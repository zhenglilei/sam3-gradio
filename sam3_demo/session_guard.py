"""Request-owner guards for Gradio callbacks.

The guard is deliberately a small adapter around :mod:`session_runtime`.
Callbacks keep their existing component arguments and receive a Gradio
``Request`` only through Gradio's special-argument injection mechanism.
"""

from __future__ import annotations

import functools
import hmac
import inspect
from copy import deepcopy
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, get_args, get_origin, get_type_hints

import gradio as gr
from gradio.context import LocalContext

from .session_runtime import (
    RequestIdentity,
    SessionError,
    SessionExpired,
    SessionRecord,
    SessionRegistry,
    resolve_client_ip,
)

if TYPE_CHECKING:
    from .observability import Observability


class SessionGuardError(SessionError):
    """Raised before a callback is called when request ownership is invalid."""


class _SupersededCallback(Exception):
    """A queued operation targets a session already replaced in this page."""


_REQUEST_PARAMETER = "__session_guard_request"
_IDENTITY_FIELDS = (
    "schema_version",
    "session_id",
    "generation",
    "owner_token",
    "owner_id_digest",
    "session_hash_digest",
    "client_ip_digest",
    "resume_id",
)
_MISSING = object()


def _request_type(annotation: Any) -> bool:
    """Return whether *annotation* is the Gradio Request special type."""

    if annotation is gr.Request:
        return True
    origin = get_origin(annotation)
    if origin is not None and type(None) in get_args(annotation):
        return any(argument is gr.Request for argument in get_args(annotation))
    return False


def _request_parameter_names(fn: Callable[..., Any]) -> set[str]:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    names: set[str] = set()
    for parameter in inspect.signature(fn).parameters.values():
        annotation = hints.get(parameter.name, parameter.annotation)
        if _request_type(annotation):
            names.add(parameter.name)
    return names


def _peer_host(request: Any) -> str:
    try:
        client = getattr(request, "client")
    except (AttributeError, TypeError) as exc:
        raise SessionGuardError("request client address is required") from exc

    if isinstance(client, Mapping):
        host = client.get("host")
    elif isinstance(client, (tuple, list)):
        host = client[0] if client else None
    else:
        host = getattr(client, "host", None)
    if not isinstance(host, str) or not host.strip():
        raise SessionGuardError("request client address is required")
    return host


def request_identity(
    request: Any,
    trusted_proxy_cidrs: Optional[Sequence[str] | str] = None,
) -> RequestIdentity:
    """Extract and validate the browser/network identity from a Gradio request."""

    if request is None:
        raise SessionGuardError("Gradio Request is required")
    session_hash = getattr(request, "session_hash", None)
    if not isinstance(session_hash, str) or not session_hash.strip():
        raise SessionGuardError("request session_hash is required")
    owner_id = getattr(request, "username", None)
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise SessionGuardError("signed cookie owner is required")
    peer = _peer_host(request)
    try:
        headers = getattr(request, "headers", {})
        headers_dict = dict(headers) if headers is not None else {}
        client_ip = resolve_client_ip(peer, headers_dict, trusted_proxy_cidrs)
    except (TypeError, ValueError, AttributeError) as exc:
        raise SessionGuardError("request client identity is invalid") from exc
    return RequestIdentity(
        owner_id=owner_id,
        session_hash=session_hash,
        client_ip=client_ip,
    )


def _session_candidates(values: Sequence[Any]) -> list[tuple[Mapping[str, Any], str]]:
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        session_id = value.get("session_id")
        if session_id in (None, ""):
            continue
        if not isinstance(session_id, str) or not session_id.strip():
            raise SessionGuardError("session_id must be a non-empty string")
        candidates.append((value, session_id))
    return candidates


def _is_full_state(state: Mapping[str, Any]) -> bool:
    return "owner_token" in state and "generation" in state


def _required_state_arguments(
    parameters: Sequence[inspect.Parameter],
    values: Sequence[Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    required: list[tuple[str, Mapping[str, Any]]] = []
    for parameter, value in zip(parameters, values):
        if parameter.name != "state" and not parameter.name.endswith("_state"):
            continue
        if not isinstance(value, Mapping):
            raise SessionGuardError(
                f"{parameter.name} must be a session-bearing state mapping"
            )
        session_id = value.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise SessionGuardError(f"{parameter.name} has no server session id")
        required.append((parameter.name, value))
    return required


def _validate_state_owner_tokens(
    states: Sequence[tuple[str, Mapping[str, Any]]],
    record: SessionRecord,
) -> None:
    for name, state in states:
        if state.get("session_id") != record.session_id:
            raise SessionGuardError(f"{name} belongs to another server session")
        token = state.get("owner_token")
        if not isinstance(token, str) or not hmac.compare_digest(
            token,
            record.owner_token,
        ):
            raise SessionGuardError(f"{name} has no valid owner token")


def _stamp_owned_outputs(value: Any, record: SessionRecord) -> Any:
    if isinstance(value, dict):
        if value.get("session_id") == record.session_id:
            value = dict(value)
            value["owner_token"] = record.owner_token
            value["resume_id"] = record.resume_id
        return value
    if isinstance(value, tuple):
        return tuple(_stamp_owned_outputs(item, record) for item in value)
    if isinstance(value, list):
        return [_stamp_owned_outputs(item, record) for item in value]
    return value


def _validate_candidates(
    candidates: Sequence[tuple[Mapping[str, Any], str]],
    *,
    registry: SessionRegistry,
    identity: RequestIdentity,
) -> tuple[Mapping[str, Any] | None, str, SessionRecord]:
    session_ids = {session_id for _, session_id in candidates}
    if len(session_ids) != 1:
        raise SessionGuardError("callback contains conflicting session states")
    session_id = next(iter(session_ids))
    full_states = [state for state, _ in candidates if _is_full_state(state)]
    if full_states:
        record: SessionRecord | None = None
        for state in full_states:
            checked = registry.validate(state, identity)
            if checked.session_id != session_id:
                raise SessionGuardError("session state does not match record")
            record = checked
        assert record is not None
        return full_states[0], session_id, record
    record = registry.validate_owner(session_id, identity)
    if record.session_id != session_id:
        raise SessionGuardError("session state does not match record")
    return None, session_id, record


def _lease_for(
    registry: SessionRegistry,
    full_state: Mapping[str, Any] | None,
    session_id: str,
    identity: RequestIdentity,
):
    if full_state is not None:
        lease = getattr(registry, "lease", None)
        if lease is None:
            raise SessionGuardError("session registry does not support state leases")
        return lease(full_state, identity)
    owner_lease = getattr(registry, "owner_lease", None)
    if owner_lease is None:
        raise SessionGuardError("session registry does not support owner leases")
    return owner_lease(session_id, identity)


def gradio_state_recovery(demo, refs, *, registry, factory):
    """Persist recovery to all owned Gradio States, not just callback arguments.

    Called under the replacement session's operation lease. Aliases in the
    factory resolve to the same canonical component value.
    """
    components = {
        name: component
        for group in refs for name, component in vars(group).items()
        if isinstance(component, gr.State)
    }

    def recover(server_state, request, *, previous_session_id=None):
        identity = request_identity(request)
        session = demo.state_holder[identity.session_hash]
        fresh = factory(server_state)
        resolved = {}
        updates = []
        root = session[components["session_state"]._id]
        if isinstance(root, Mapping) and root.get("session_id"):
            registry.authenticate_state(root, identity)
            if (previous_session_id is not None
                    and previous_session_id != server_state["session_id"]
                    and root["session_id"] == server_state["session_id"]):
                raise _SupersededCallback
        reset = not isinstance(root, Mapping) or root.get("session_id") != server_state["session_id"]
        for name, component in components.items():
            current = session[component._id]
            if name not in fresh:
                # Extension states (e.g. the EL image queue) reset to their own
                # component default; old image payloads are never re-authorized.
                stale_extra = (
                    isinstance(current, Mapping) and current.get("session_id")
                    and current["session_id"] != server_state["session_id"]
                )
                replacement = deepcopy(component.value) if reset or stale_extra else current
                fresh[name] = replacement
                if name.endswith("_state"):
                    fresh.setdefault(name[:-6], replacement)
                if replacement is not current:
                    updates.append((component._id, replacement))
                continue
            replacement = fresh[name]
            if isinstance(current, Mapping) and current.get("session_id"):
                registry.authenticate_state(current, identity)
                if current["session_id"] == server_state["session_id"]:
                    replacement = current
            resolved[id(fresh[name])] = replacement
            if replacement is not current:
                updates.append((component._id, replacement))
        # Validate the entire bundle before publishing any replacement.
        for component_id, replacement in updates:
            session[component_id] = replacement
        return {name: resolved.get(id(value), value) for name, value in fresh.items()}

    return recover


def guard_callback(
    fn: Callable[..., Any],
    *,
    registry: SessionRegistry,
    trusted_proxy_cidrs: Optional[Sequence[str] | str] = None,
    observability: "Observability | None" = None,
    recovery_factory: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]
    ] = None,
    recovery_store: Optional[Callable[..., Mapping[str, Mapping[str, Any]]]] = None,
) -> Callable[..., Any]:
    """Wrap a callback with strict request/session ownership validation.

    The generated signature contains one trailing optional ``gr.Request``
    parameter.  Gradio recognizes that annotation as a special argument and
    injects the request without adding a component input to the event config.
    """

    if not callable(fn):
        raise TypeError("callback must be callable")
    if not isinstance(registry, SessionRegistry):
        raise TypeError("registry must be a SessionRegistry")
    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        raise TypeError("session guard does not support variadic callbacks")
    if any(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters):
        raise TypeError("session guard requires positional callback parameters")
    if _request_parameter_names(fn):
        raise TypeError("callback already declares a Gradio Request parameter")
    if any(
        parameter.name == "request"
        or parameter.name == _REQUEST_PARAMETER
        or parameter.name.endswith(_REQUEST_PARAMETER)
        for parameter in parameters
    ):
        raise TypeError("callback parameter conflicts with session guard Request")

    request_parameter = inspect.Parameter(
        _REQUEST_PARAMETER,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None,
        annotation=gr.Request,
    )
    guarded_signature = signature.replace(parameters=parameters + [request_parameter])

    def _guarded_impl(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise SessionGuardError("guarded callbacks accept positional inputs only")
        expected = len(parameters) + 1
        if len(args) != expected:
            raise SessionGuardError(
                f"guarded callback expected {len(parameters)} component inputs and a Request"
            )
        component_args = list(args[:-1])
        request = args[-1]
        identity = request_identity(request, trusted_proxy_cidrs)
        state_positions = [
            (index, parameter.name, component_args[index])
            for index, parameter in enumerate(parameters)
            if parameter.name == "state" or parameter.name.endswith("_state")
            or (isinstance(component_args[index], Mapping)
                and component_args[index].get("session_id"))
        ]
        missing_state = any(
            value is None
            or (isinstance(value, Mapping) and not value.get("session_id"))
            for _, _, value in state_positions
        )
        candidates = _session_candidates(component_args)
        full_states = [state for state, _ in candidates if _is_full_state(state)]
        server_state = None
        recovered = False
        try:
            if recovery_factory is not None:
                if any(value is not None and not isinstance(value, Mapping)
                       for _, _, value in state_positions):
                    raise SessionGuardError("state values must be mappings")
                if len({session_id for _, session_id in candidates}) > 1:
                    raise SessionGuardError("callback contains conflicting session states")
                stale_state = False
                if candidates:
                    try:
                        _validate_candidates(candidates, registry=registry, identity=identity)
                    except SessionExpired:
                        stale_state = True
                if missing_state or stale_state:
                    # Even a thin business state must prove ownership before
                    # recovery. A resume key alone is not an authorization token.
                    for state in full_states:
                        registry.authenticate_state(state, identity)
                    for _, name, state in state_positions:
                        if not isinstance(state, Mapping) or not state.get("session_id"):
                            continue
                        if (full_states and "owner_token" not in state
                                and name != "state" and not name.endswith("_state")):
                            continue  # Legacy extension handle is discarded below.
                        registry.authenticate_state(state, identity)
                    server_state, recovered = registry.ensure(
                        full_states[0] if full_states else
                        (candidates[0][0] if candidates else None), identity,
                    )
            recovery_lease = (
                registry.lease(server_state, identity)
                if server_state is not None else nullcontext()
            )
            with recovery_lease:
                if server_state is not None:
                    # Gradio 6 propagates the owning Blocks through this request
                    # context, including callbacks registered by extension tabs.
                    blocks = (
                        LocalContext.blocks.get(None)
                        if LocalContext.request.get(None) is request else None
                    )
                    store = recovery_store or getattr(blocks, "_sam3_session_recovery", None)
                    fresh_states = (
                        store(
                            server_state, request,
                            previous_session_id=candidates[0][1] if recovered and candidates else None,
                        ) if store is not None
                        else recovery_factory(server_state)
                    )
                    for index, name, value in state_positions:
                        if recovered or value is None or not value.get("session_id"):
                            replacement = fresh_states.get(name, _MISSING)
                            if (replacement is _MISSING or
                                ((name == "state" or name.endswith("_state"))
                                 and not isinstance(replacement, Mapping))):
                                raise SessionGuardError(f"recovery factory did not provide {name}")
                            component_args[index] = replacement
                return invoke(component_args, identity)
        except _SupersededCallback:
            # Gradio broadcasts a single skip to all outputs. Never replay an
            # expired delete/crop action against a newly recovered workspace.
            return gr.skip()
        except SessionGuardError:
            raise
        except SessionError as exc:
            raise SessionGuardError(str(exc)) from exc

    def invoke(component_args, identity):
        required_states = _required_state_arguments(parameters, component_args)
        candidates = _session_candidates(component_args)
        if not candidates:
            raise SessionGuardError("guarded callback requires a session-bearing state")
        full_state, session_id, record = _validate_candidates(
            candidates, registry=registry, identity=identity,
        )
        _validate_state_owner_tokens(required_states, record)
        with _lease_for(registry, full_state, session_id, identity):
            identity_snapshots = [
                (
                    state,
                    {field: state.get(field, _MISSING) for field in _IDENTITY_FIELDS},
                )
                for _, state in required_states
                if isinstance(state, dict)
            ]
            try:
                result = fn(*component_args)
            except BaseException:
                # Gradio State inputs are server-side mutable dicts. Preserve
                # ownership even if a callback mutates in place before raising.
                for state, snapshot in identity_snapshots:
                    for field, value in snapshot.items():
                        if value is _MISSING:
                            state.pop(field, None)
                        else:
                            state[field] = value
                raise
            if full_state is not None:
                record = registry.validate(full_state, identity)
            else:
                record = registry.validate_owner(session_id, identity)
            if record.session_id != session_id:
                raise SessionGuardError("session closed or changed during callback")
            return _stamp_owned_outputs(result, record)

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if observability is None:
            return _guarded_impl(*args, **kwargs)
        started_at = observability.monotonic()
        request = args[-1] if args else None
        owner_id = getattr(request, "username", None)
        session_hash = getattr(request, "session_hash", None)
        action = fn.__name__.lstrip("_") or fn.__name__
        try:
            result = _guarded_impl(*args, **kwargs)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            error_id = observability.record_callback_failure(
                action,
                observability.monotonic() - started_at,
                exc,
                owner_id=owner_id,
                session_hash=session_hash,
            )
            if isinstance(exc, SessionGuardError):
                raise SessionGuardError(f"{exc}（错误编号：{error_id}）") from exc
            if isinstance(exc, gr.Error):
                message = getattr(exc, "message", None) or "操作失败"
                raise gr.Error(f"{message}（错误编号：{error_id}）") from exc
            raise gr.Error(f"操作失败，请联系开发人员。错误编号：{error_id}") from exc
        observability.record_callback_success(
            action,
            observability.monotonic() - started_at,
            owner_id=owner_id,
            session_hash=session_hash,
        )
        return result

    guarded.__signature__ = guarded_signature
    annotations = dict(getattr(guarded, "__annotations__", {}))
    annotations[_REQUEST_PARAMETER] = gr.Request
    guarded.__annotations__ = annotations
    return guarded


__all__ = ["SessionGuardError", "guard_callback", "gradio_state_recovery", "request_identity"]
