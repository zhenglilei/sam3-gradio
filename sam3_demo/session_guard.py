"""Request-owner guards for Gradio callbacks.

The guard is deliberately a small adapter around :mod:`session_runtime`.
Callbacks keep their existing component arguments and receive a Gradio
``Request`` only through Gradio's special-argument injection mechanism.
"""

from __future__ import annotations

import functools
import hmac
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, get_args, get_origin, get_type_hints

import gradio as gr

from .session_runtime import (
    SessionError,
    SessionExpired,
    SessionRecord,
    SessionRegistry,
    resolve_client_ip,
)


class SessionGuardError(SessionError):
    """Raised before a callback is called when request ownership is invalid."""


_REQUEST_PARAMETER = "__session_guard_request"
_IDENTITY_FIELDS = (
    "schema_version",
    "session_id",
    "generation",
    "owner_token",
    "session_hash_digest",
    "client_ip_digest",
    "resume_id",
)
_MISSING = object()
_BUSINESS_STATE_MARKERS = frozenset(
    {
        "image_id",
        "source_image_id",
        "instances",
        "bbox_start",
        "layout_id",
        "regions_revision",
        "conversation_revision",
    }
)


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
) -> tuple[str, str]:
    """Extract and validate the browser/network identity from a Gradio request."""

    if request is None:
        raise SessionGuardError("Gradio Request is required")
    session_hash = getattr(request, "session_hash", None)
    if not isinstance(session_hash, str) or not session_hash.strip():
        raise SessionGuardError("request session_hash is required")
    peer = _peer_host(request)
    try:
        headers = getattr(request, "headers", {})
        headers_dict = dict(headers) if headers is not None else {}
        client_ip = resolve_client_ip(peer, headers_dict, trusted_proxy_cidrs)
    except (TypeError, ValueError, AttributeError) as exc:
        raise SessionGuardError("request client identity is invalid") from exc
    return session_hash, client_ip


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
        if (
            value.get("session_id") == record.session_id
            and (
                "owner_token" in value
                or any(marker in value for marker in _BUSINESS_STATE_MARKERS)
            )
        ):
            value = dict(value)
            value["owner_token"] = record.owner_token
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
    session_hash: str,
    client_ip: str,
) -> tuple[Mapping[str, Any] | None, str, SessionRecord]:
    session_ids = {session_id for _, session_id in candidates}
    if len(session_ids) != 1:
        raise SessionGuardError("callback contains conflicting session states")
    session_id = next(iter(session_ids))
    full_states = [state for state, _ in candidates if _is_full_state(state)]
    if full_states:
        record: SessionRecord | None = None
        for state in full_states:
            checked = registry.validate(state, session_hash, client_ip)
            if checked.session_id != session_id:
                raise SessionGuardError("session state does not match record")
            record = checked
        assert record is not None
        return full_states[0], session_id, record
    record = registry.validate_owner(session_id, session_hash, client_ip)
    if record.session_id != session_id:
        raise SessionGuardError("session state does not match record")
    return None, session_id, record


def _lease_for(
    registry: SessionRegistry,
    full_state: Mapping[str, Any] | None,
    session_id: str,
    session_hash: str,
    client_ip: str,
):
    if full_state is not None:
        lease = getattr(registry, "lease", None)
        if lease is None:
            raise SessionGuardError("session registry does not support state leases")
        return lease(full_state, session_hash, client_ip)
    owner_lease = getattr(registry, "owner_lease", None)
    if owner_lease is None:
        raise SessionGuardError("session registry does not support owner leases")
    return owner_lease(session_id, session_hash, client_ip)


def guard_callback(
    fn: Callable[..., Any],
    *,
    registry: SessionRegistry,
    trusted_proxy_cidrs: Optional[Sequence[str] | str] = None,
    recovery_factory: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]
    ] = None,
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

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise SessionGuardError("guarded callbacks accept positional inputs only")
        expected = len(parameters) + 1
        if len(args) != expected:
            raise SessionGuardError(
                f"guarded callback expected {len(parameters)} component inputs and a Request"
            )
        component_args = list(args[:-1])
        request = args[-1]
        session_hash, client_ip = request_identity(request, trusted_proxy_cidrs)
        state_positions = [
            (index, parameter.name, component_args[index])
            for index, parameter in enumerate(parameters)
            if parameter.name == "state" or parameter.name.endswith("_state")
        ]
        missing_state = any(
            value is None
            or (isinstance(value, Mapping) and not value.get("session_id"))
            for _, _, value in state_positions
        )
        candidates = _session_candidates(component_args)
        full_states = [state for state, _ in candidates if _is_full_state(state)]
        stale_state = False
        if recovery_factory is not None and not missing_state and full_states:
            try:
                registry.validate(full_states[0], session_hash, client_ip)
            except SessionExpired:
                stale_state = True
            except SessionError:
                pass
        if (missing_state or stale_state) and recovery_factory is not None:
            if any(
                value is not None and not isinstance(value, Mapping)
                for _, _, value in state_positions
            ):
                raise SessionGuardError("state values must be mappings")
            session_ids = {session_id for _, session_id in candidates}
            if len(session_ids) > 1:
                raise SessionGuardError("callback contains conflicting session states")
            if candidates and not full_states:
                raise SessionGuardError("stale callback has no recoverable session state")
            try:
                server_state, recovered = registry.ensure(
                    full_states[0] if full_states else None,
                    session_hash,
                    client_ip,
                )
            except SessionError as exc:
                raise SessionGuardError(str(exc)) from exc
            fresh_states = recovery_factory(server_state)
            for index, name, value in state_positions:
                if recovered or value is None or not value.get("session_id"):
                    replacement = fresh_states.get(name)
                    if not isinstance(replacement, Mapping):
                        raise SessionGuardError(
                            f"recovery factory did not provide {name}"
                        )
                    component_args[index] = replacement
        required_states = _required_state_arguments(parameters, component_args)
        candidates = _session_candidates(component_args)
        if not candidates:
            raise SessionGuardError("guarded callback requires a session-bearing state")
        try:
            full_state, session_id, record = _validate_candidates(
                candidates,
                registry=registry,
                session_hash=session_hash,
                client_ip=client_ip,
            )
            _validate_state_owner_tokens(required_states, record)
            with _lease_for(registry, full_state, session_id, session_hash, client_ip):
                identity_snapshots = [
                    (
                        state,
                        {
                            field: state.get(field, _MISSING)
                            for field in _IDENTITY_FIELDS
                        },
                    )
                    for _, state in required_states
                    if isinstance(state, dict)
                ]
                try:
                    result = fn(*component_args)
                except BaseException:
                    # Gradio State inputs are server-side mutable dicts.  Preserve
                    # their ownership identity even if a callback mutates in place
                    # before raising, otherwise one failure can brick the session.
                    for state, snapshot in identity_snapshots:
                        for field, value in snapshot.items():
                            if value is _MISSING:
                                state.pop(field, None)
                            else:
                                state[field] = value
                    raise
                if full_state is not None:
                    record = registry.validate(full_state, session_hash, client_ip)
                else:
                    record = registry.validate_owner(session_id, session_hash, client_ip)
                if record.session_id != session_id:
                    raise SessionGuardError("session closed or changed during callback")
                return _stamp_owned_outputs(result, record)
        except SessionGuardError:
            raise
        except SessionError as exc:
            raise SessionGuardError(str(exc)) from exc

    guarded.__signature__ = guarded_signature
    annotations = dict(getattr(guarded, "__annotations__", {}))
    annotations[_REQUEST_PARAMETER] = gr.Request
    guarded.__annotations__ = annotations
    return guarded


__all__ = ["SessionGuardError", "guard_callback", "request_identity"]
