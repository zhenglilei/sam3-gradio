const or = 1, cr = 2, hr = 16, _r = 1, dr = 2, vr = 4, pr = 8, yr = 16, gr = 1, wr = 2, S = /* @__PURE__ */ Symbol("uninitialized"), tn = "http://www.w3.org/1999/xhtml", mr = "http://www.w3.org/2000/svg", Er = "@attach", gt = !1;
var nn = Array.isArray, rn = Array.prototype.indexOf, Me = Array.prototype.includes, sn = Array.from, ln = Object.defineProperty, we = Object.getOwnPropertyDescriptor, fn = Object.getOwnPropertyDescriptors, an = Object.prototype, un = Array.prototype, wt = Object.getPrototypeOf, ot = Object.isExtensible;
function br(e) {
  return typeof e == "function";
}
const on = () => {
};
function Sr(e) {
  return e();
}
function cn(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function mt() {
  var e, t, n = new Promise((r, s) => {
    e = r, t = s;
  });
  return { promise: n, resolve: e, reject: t };
}
const T = 2, ce = 4, Te = 8, et = 1 << 24, M = 16, L = 32, Y = 64, $e = 128, C = 512, E = 1024, b = 2048, R = 4096, O = 8192, F = 16384, _e = 32768, ct = 1 << 25, be = 65536, Ne = 1 << 17, hn = 1 << 18, de = 1 << 19, Et = 1 << 20, Tr = 1 << 25, ie = 65536, De = 1 << 21, oe = 1 << 22, Z = 1 << 23, re = /* @__PURE__ */ Symbol("$state"), kr = /* @__PURE__ */ Symbol("legacy props"), Ar = /* @__PURE__ */ Symbol(""), _n = /* @__PURE__ */ Symbol("attributes"), dn = /* @__PURE__ */ Symbol("class"), vn = /* @__PURE__ */ Symbol("style"), ze = /* @__PURE__ */ Symbol("text"), ke = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}(), Cr = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  !!globalThis.document?.contentType && /* @__PURE__ */ globalThis.document.contentType.includes("xml")
);
function pn() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function Rr(e, t, n) {
  throw new Error("https://svelte.dev/e/each_key_duplicate");
}
function yn(e) {
  throw new Error("https://svelte.dev/e/effect_in_teardown");
}
function gn() {
  throw new Error("https://svelte.dev/e/effect_in_unowned_derived");
}
function wn(e) {
  throw new Error("https://svelte.dev/e/effect_orphan");
}
function mn() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function Or(e) {
  throw new Error("https://svelte.dev/e/props_invalid_value");
}
function En() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function bn() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function Sn() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function Tn() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function kn() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function Pr() {
  console.warn("https://svelte.dev/e/select_multiple_invalid_value");
}
function An() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function bt(e) {
  return e === this.v;
}
function xn(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function St(e) {
  return !xn(e, this.v);
}
let Be = !1, Cn = !1;
function Ir() {
  Be = !0;
}
let m = null;
function he(e) {
  m = e;
}
function Rn(e, t = !1, n) {
  m = {
    p: m,
    i: !1,
    c: null,
    e: null,
    s: e,
    x: null,
    r: (
      /** @type {Effect} */
      p
    ),
    l: Be && !t ? { s: null, u: null, $: [] } : null
  };
}
function On(e) {
  var t = (
    /** @type {ComponentContext} */
    m
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      Ht(r);
  }
  return t.i = !0, m = t.p, /** @type {T} */
  {};
}
function Ae() {
  return !Be || m !== null && m.l === null;
}
let te = [];
function Tt() {
  var e = te;
  te = [], cn(e);
}
function K(e) {
  if (te.length === 0 && !me) {
    var t = te;
    queueMicrotask(() => {
      t === te && Tt();
    });
  }
  te.push(e);
}
function Pn() {
  for (; te.length > 0; )
    Tt();
}
function kt(e) {
  var t = p;
  if (t === null)
    return v.f |= Z, e;
  if ((t.f & _e) === 0 && (t.f & ce) === 0)
    throw e;
  W(e, t);
}
function W(e, t) {
  if (!(t !== null && (t.f & F) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & $e) !== 0) {
        if ((t.f & _e) === 0)
          throw e;
        try {
          t.b.error(e);
          return;
        } catch (n) {
          e = n;
        }
      }
      t = t.parent;
    }
    throw e;
  }
}
const In = -7169;
function w(e, t) {
  e.f = e.f & In | t;
}
function tt(e) {
  (e.f & C) !== 0 || e.deps === null ? w(e, E) : w(e, R);
}
function At(e) {
  if (e !== null)
    for (const t of e)
      (t.f & T) === 0 || (t.f & ie) === 0 || (t.f ^= ie, At(
        /** @type {Derived} */
        t.deps
      ));
}
function xt(e, t, n) {
  (e.f & b) !== 0 ? t.add(e) : (e.f & R) !== 0 && n.add(e), At(e.deps), w(e, E);
}
function xe(e) {
  var t = v, n = p;
  P(null), I(null);
  try {
    return e();
  } finally {
    P(t), I(n);
  }
}
function Mn(e) {
  let t = 0, n = Ce(0), r;
  return () => {
    lt() && (X(n), ft(() => (t === 0 && (r = ut(() => e(() => Ee(n)))), t += 1, () => {
      K(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, Ee(n));
      });
    })));
  };
}
var Nn = be | de;
function Dn(e, t, n, r) {
  new Fn(e, t, n, r);
}
class Fn {
  /** @type {Boundary | null} */
  parent;
  is_pending = !1;
  /**
   * API-level transformError transform function. Transforms errors before they reach the `failed` snippet.
   * Inherited from parent boundary, or defaults to identity.
   * @type {(error: unknown) => unknown}
   */
  transform_error;
  /** @type {TemplateNode} */
  #i;
  /** @type {TemplateNode | null} */
  #v = null;
  /** @type {BoundaryProps} */
  #r;
  /** @type {((anchor: Node) => void)} */
  #c;
  /** @type {Effect} */
  #n;
  /** @type {Effect | null} */
  #l = null;
  /** @type {Effect | null} */
  #e = null;
  /** @type {Effect | null} */
  #s = null;
  /** @type {DocumentFragment | null} */
  #t = null;
  #d = 0;
  #f = 0;
  #a = !1;
  /** @type {Set<Effect>} */
  #o = /* @__PURE__ */ new Set();
  /** @type {Set<Effect>} */
  #p = /* @__PURE__ */ new Set();
  /**
   * A source containing the number of pending async deriveds/expressions.
   * Only created if `$effect.pending()` is used inside the boundary,
   * otherwise updating the source results in needless `Batch.ensure()`
   * calls followed by no-op flushes
   * @type {Source<number> | null}
   */
  #u = null;
  #g = Mn(() => (this.#u = Ce(this.#d), () => {
    this.#u = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(t, n, r, s) {
    this.#i = t, this.#r = n, this.#c = (i) => {
      var a = (
        /** @type {Effect} */
        p
      );
      a.b = this, a.f |= $e, r(i);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = s ?? this.parent?.transform_error ?? ((i) => i), this.#n = Jn(() => {
      this.#h();
    }, Nn);
  }
  #y() {
    try {
      this.#l = ee(() => this.#c(this.#i));
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #E(t) {
    const n = this.#r.failed, { reset: r, invoke_onerror: s } = this.#w(t);
    K(s), n && (this.#s = ee(() => {
      n(
        this.#i,
        () => t,
        () => r
      );
    }));
  }
  /**
   * Creates the `reset` function for a failed boundary, along with a function
   * that invokes `onerror` with it (if provided)
   * @param {unknown} error
   * @returns {{ reset: () => void, invoke_onerror: () => void }}
   */
  #w(t) {
    var n = !1, r = !1;
    const s = () => {
      if (n) {
        An();
        return;
      }
      n = !0, r && Tn(), this.#s !== null && Pe(this.#s, () => {
        this.#s = null;
      }), this.#_(() => {
        this.#h();
      });
    };
    return { reset: s, invoke_onerror: () => {
      try {
        r = !0, this.#r.onerror?.(t, s), r = !1;
      } catch (a) {
        W(a, this.#n && this.#n.parent);
      }
    } };
  }
  #b() {
    const t = this.#r.pending;
    t && (this.is_pending = !0, this.#e = ee(() => t(this.#i)), K(() => {
      var n = this.#t = document.createDocumentFragment(), r = jt();
      n.append(r), this.#l = this.#_(() => ee(() => this.#c(r))), this.#f === 0 && (this.#i.before(n), this.#t = null, Pe(
        /** @type {Effect} */
        this.#e,
        () => {
          this.#e = null;
        }
      ), this.#m(
        /** @type {Batch} */
        y
      ));
    }));
  }
  #h() {
    try {
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#d = 0, this.#l = ee(() => {
        this.#c(this.#i);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        tr(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = ee(() => n(this.#i));
      } else
        this.#m(
          /** @type {Batch} */
          y
        );
    } catch (n) {
      this.error(n);
    }
  }
  /**
   * @param {Batch} batch
   */
  #m(t) {
    this.is_pending = !1, t.transfer_effects(this.#o, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(t) {
    xt(t, this.#o, this.#p);
  }
  /**
   * Returns `false` if the effect exists inside a boundary whose pending snippet is shown
   * @returns {boolean}
   */
  is_rendered() {
    return !this.is_pending && (!this.parent || this.parent.is_rendered());
  }
  has_pending_snippet() {
    return !!this.#r.pending;
  }
  /**
   * @template T
   * @param {() => T} fn
   */
  #_(t) {
    var n = p, r = v, s = m;
    I(this.#n), P(this.#n), he(this.#n.ctx);
    try {
      return J.ensure(), t();
    } catch (i) {
      return kt(i), null;
    } finally {
      I(n), P(r), he(s);
    }
  }
  /**
   * Updates the pending count associated with the currently visible pending snippet,
   * if any, such that we can replace the snippet with content once work is done
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  #S(t, n) {
    if (!this.has_pending_snippet()) {
      this.parent && this.parent.#S(t, n);
      return;
    }
    this.#f += t, this.#f === 0 && (this.#m(n), this.#e && Pe(this.#e, () => {
      this.#e = null;
    }), this.#t && (this.#i.before(this.#t), this.#t = null));
  }
  /**
   * Update the source that powers `$effect.pending()` inside this boundary,
   * and controls when the current `pending` snippet (if any) is removed.
   * Do not call from inside the class
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  update_pending_count(t, n) {
    this.#S(t, n), this.#d += t, !(!this.#u || this.#a) && (this.#a = !0, K(() => {
      this.#a = !1, this.#u && je(this.#u, this.#d);
    }));
  }
  get_effect_pending() {
    return this.#g(), X(
      /** @type {Source<number>} */
      this.#u
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    y?.is_fork ? (this.#l && y.skip_effect(this.#l), this.#e && y.skip_effect(this.#e), this.#s && y.skip_effect(this.#s), y.oncommit(() => {
      this.#T(t);
    })) : this.#T(t);
  }
  /**
   * @param {unknown} error
   */
  #T(t) {
    this.#l && (V(this.#l), this.#l = null), this.#e && (V(this.#e), this.#e = null), this.#s && (V(this.#s), this.#s = null);
    let n = this.#r.failed;
    const r = (s) => {
      const { reset: i, invoke_onerror: a } = this.#w(s);
      a(), n && (this.#s = this.#_(() => {
        try {
          return ee(() => {
            var f = (
              /** @type {Effect} */
              p
            );
            f.b = this, f.f |= $e, n(
              this.#i,
              () => s,
              () => i
            );
          });
        } catch (f) {
          return W(
            f,
            /** @type {Effect} */
            this.#n.parent
          ), null;
        }
      }));
    };
    K(() => {
      var s;
      try {
        s = this.transform_error(t);
      } catch (i) {
        W(i, this.#n && this.#n.parent);
        return;
      }
      s !== null && typeof s == "object" && typeof /** @type {any} */
      s.then == "function" ? s.then(
        r,
        /** @param {unknown} e */
        (i) => W(i, this.#n && this.#n.parent)
      ) : r(s);
    });
  }
}
function Ln(e, t, n, r) {
  const s = Ae() ? nt : Vn;
  var i = e.filter((h) => !h.settled), a = t.map(s);
  if (n.length === 0 && i.length === 0) {
    r(a);
    return;
  }
  var f = (
    /** @type {Effect} */
    p
  ), l = jn(), u = i.length === 1 ? i[0].promise : i.length > 1 ? Promise.all(i.map((h) => h.promise)) : null;
  function _(h) {
    if ((f.f & F) === 0) {
      l();
      try {
        r([...a, ...h]);
      } catch (d) {
        W(d, f);
      }
      Fe();
    }
  }
  var o = Ct();
  if (n.length === 0) {
    u.then(() => _([])).finally(o);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ Bn(h))).then(_).catch((h) => W(h, f)).finally(o);
  }
  u ? u.then(() => {
    l(), c(), Fe();
  }) : c();
}
function jn() {
  var e = (
    /** @type {Effect} */
    p
  ), t = v, n = m, r = (
    /** @type {Batch} */
    y
  );
  return function(i = !0) {
    I(e), P(t), he(n), i && (e.f & F) === 0 && (r?.activate(), r?.apply());
  };
}
function Fe(e = !0) {
  I(null), P(null), he(null), e && y?.deactivate();
}
function Ct() {
  var e = (
    /** @type {Effect} */
    p
  ), t = e.b, n = (
    /** @type {Batch} */
    y
  ), r = !!t?.is_rendered();
  return t?.update_pending_count(1, n), n.increment(r, e), () => {
    t?.update_pending_count(-1, n), n.decrement(r, e);
  };
}
// @__NO_SIDE_EFFECTS__
function nt(e) {
  var t = T | b;
  return p !== null && (p.f |= de), {
    ctx: m,
    deps: null,
    effects: null,
    equals: bt,
    f: t,
    fn: e,
    reactions: null,
    rv: 0,
    v: (
      /** @type {V} */
      S
    ),
    wv: 0,
    parent: p,
    ac: null
  };
}
const pe = /* @__PURE__ */ Symbol("obsolete");
// @__NO_SIDE_EFFECTS__
function Bn(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && pn();
  var s = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), i = Ce(
    /** @type {V} */
    S
  ), a = !v, f = /* @__PURE__ */ new Set();
  return Zn(() => {
    var l = (
      /** @type {Effect} */
      p
    ), u = mt();
    s = u.promise;
    try {
      Promise.resolve(e()).then(u.resolve, (h) => {
        h !== ke && u.reject(h);
      }).finally(Fe);
    } catch (h) {
      u.reject(h), Fe();
    }
    var _ = (
      /** @type {Batch} */
      y
    );
    if (a) {
      if ((l.f & _e) !== 0)
        var o = Ct();
      if (
        // boundary can be null if the async derived is inside an $effect.root not connected to the component render tree
        r.b?.is_rendered()
      )
        _.async_deriveds.get(l)?.reject(pe);
      else
        for (const h of f.values())
          h.reject(pe);
      f.add(u), _.async_deriveds.set(l, u);
    }
    const c = (h, d = void 0) => {
      o?.(), f.delete(u), d !== pe && (_.activate(), d ? (i.f |= Z, je(i, d)) : ((i.f & Z) !== 0 && (i.f ^= Z), je(i, h)), _.deactivate());
    };
    u.promise.then(c, (h) => c(null, h || "unknown"));
  }), qt(() => {
    for (const l of f)
      l.reject(pe);
  }), new Promise((l) => {
    function u(_) {
      function o() {
        _ === s ? l(i) : u(s);
      }
      _.then(o, o);
    }
    u(s);
  });
}
// @__NO_SIDE_EFFECTS__
function Mr(e) {
  const t = /* @__PURE__ */ nt(e);
  return zt(t), t;
}
// @__NO_SIDE_EFFECTS__
function Vn(e) {
  const t = /* @__PURE__ */ nt(e);
  return t.equals = St, t;
}
function qn(e) {
  var t = e.effects;
  if (t !== null) {
    e.effects = null;
    for (var n = 0; n < t.length; n += 1)
      V(
        /** @type {Effect} */
        t[n]
      );
  }
}
function rt(e) {
  var t, n = p, r = e.parent;
  if (!Q && r !== null && e.v !== S && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (F | O)) !== 0)
    return kn(), e.v;
  I(r);
  try {
    e.f &= ~ie, qn(e), t = Zt(e);
  } finally {
    I(n);
  }
  return t;
}
function Rt(e) {
  var t = rt(e);
  if (!e.equals(t) && (e.wv = Wt(), (!y?.is_fork || e.deps === null) && (y !== null ? (y.capture(e, t, !0), Ke?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, E);
    return;
  }
  Q || (N !== null ? (lt() || y?.is_fork) && N.set(e, t) : tt(e));
}
function Hn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac !== null && xe(() => {
        t.ac.abort(ke), t.ac = null;
      }), t.fn !== null && (t.teardown = on), Se(t, 0), at(t));
}
function Ot(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && le(t);
}
let He = null, ae = null, y = null, Ke = null, N = null, We = null, me = !1, Ue = !1, ue = null, Oe = null;
var ht = 0;
let Un = 1;
class J {
  id = Un++;
  /** True as soon as `#process` was called */
  #i = !1;
  linked = !0;
  /** @type {Batch | null} */
  #v = null;
  /** @type {Batch | null} */
  #r = null;
  /** @type {Map<Effect, ReturnType<typeof deferred<any>>>} */
  async_deriveds = /* @__PURE__ */ new Map();
  /**
   * The current values of any signals that are updated in this batch.
   * Tuple format: [value, is_derived] (note: is_derived is false for deriveds, too, if they were overridden via assignment)
   * They keys of this map are identical to `this.#previous`
   * @type {Map<Value, [any, boolean]>}
   */
  current = /* @__PURE__ */ new Map();
  /**
   * The values of any signals (sources and deriveds) that are updated in this batch _before_ those updates took place.
   * They keys of this map are identical to `this.#current`
   * @type {Map<Value, any>}
   */
  previous = /* @__PURE__ */ new Map();
  /**
   * When the batch is committed (and the DOM is updated), we need to remove old branches
   * and append new ones by calling the functions added inside (if/each/key/etc) blocks
   * @type {Set<(batch: Batch) => void>}
   */
  #c = /* @__PURE__ */ new Set();
  /**
   * If a fork is discarded, we need to destroy any effects that are no longer needed
   * @type {Set<(batch: Batch) => void>}
   */
  #n = /* @__PURE__ */ new Set();
  /**
   * The number of async effects that are currently in flight
   */
  #l = 0;
  /**
   * Async effects that are currently in flight, _not_ inside a pending boundary
   * @type {Map<Effect, number>}
   */
  #e = /* @__PURE__ */ new Map();
  /**
   * A deferred that resolves when the batch is committed, used with `settled()`
   * TODO replace with Promise.withResolvers once supported widely enough
   * @type {{ promise: Promise<void>, resolve: (value?: any) => void, reject: (reason: unknown) => void } | null}
   */
  #s = null;
  /**
   * The root effects that need to be flushed
   * @type {Effect[]}
   */
  #t = [];
  /**
   * Effects created while this batch was active.
   * @type {Effect[]}
   */
  #d = [];
  /**
   * Deferred effects (which run after async work has completed) that are DIRTY
   * @type {Set<Effect>}
   */
  #f = /* @__PURE__ */ new Set();
  /**
   * Deferred effects that are MAYBE_DIRTY
   * @type {Set<Effect>}
   */
  #a = /* @__PURE__ */ new Set();
  /**
   * A map of branches that still exist, but will be destroyed when this batch
   * is committed — we skip over these during `process`.
   * The value contains child effects that were dirty/maybe_dirty before being reset,
   * so they can be rescheduled if the branch survives.
   * @type {Map<Effect, { d: Effect[], m: Effect[] }>}
   */
  #o = /* @__PURE__ */ new Map();
  /**
   * Inverse of #skipped_branches which we need to tell prior batches to unskip them when committing
   * @type {Set<Effect>}
   */
  #p = /* @__PURE__ */ new Set();
  is_fork = !1;
  #u = !1;
  constructor() {
    ae === null ? He = ae = this : (ae.#r = this, this.#v = ae), ae = this;
  }
  #g() {
    if (this.is_fork) return !0;
    for (const r of this.#e.keys()) {
      for (var t = r, n = !1; t.parent !== null; ) {
        if (this.#o.has(t)) {
          n = !0;
          break;
        }
        t = t.parent;
      }
      if (!n)
        return !0;
    }
    return !1;
  }
  /**
   * Add an effect to the #skipped_branches map and reset its children
   * @param {Effect} effect
   */
  skip_effect(t) {
    this.#o.has(t) || this.#o.set(t, { d: [], m: [] }), this.#p.delete(t);
  }
  /**
   * Remove an effect from the #skipped_branches map and reschedule
   * any tracked dirty/maybe_dirty child effects
   * @param {Effect} effect
   * @param {(e: Effect) => void} callback
   */
  unskip_effect(t, n = (r) => this.schedule(r)) {
    var r = this.#o.get(t);
    if (r) {
      this.#o.delete(t);
      for (var s of r.d)
        w(s, b), n(s);
      for (s of r.m)
        w(s, R), n(s);
    }
    this.#p.add(t);
  }
  #y() {
    this.#i = !0, ht++ > 1e3 && (this.#_(), Gn());
    for (const l of this.#f)
      this.#a.delete(l), w(l, b), this.schedule(l);
    for (const l of this.#a)
      w(l, R), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = ue = [], r = [], s = Oe = [];
    for (const l of t)
      try {
        this.#E(l, n, r);
      } catch (u) {
        throw Mt(l), this.#g() || this.discard(), u;
      }
    if (y = null, s.length > 0) {
      var i = J.ensure();
      for (const l of s)
        i.schedule(l);
    }
    if (ue = null, Oe = null, this.#g()) {
      this.#h(r), this.#h(n);
      for (const [l, u] of this.#o)
        It(l, u);
      s.length > 0 && /** @type {unknown} */
      y.#y();
      return;
    }
    const a = this.#w();
    if (a) {
      this.#h(r), this.#h(n), a.#b(this);
      return;
    }
    this.#f.clear(), this.#a.clear();
    for (const l of this.#c) l(this);
    this.#c.clear(), Ke = this, _t(r), _t(n), Ke = null, this.#s?.resolve();
    var f = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      y
    );
    if (this.#l === 0 && (this.#t.length === 0 || f !== null) && this.#_(), this.#t.length > 0)
      if (f !== null) {
        const l = f;
        l.#t.push(...this.#t.filter((u) => !l.#t.includes(u)));
      } else
        f = this;
    f !== null && (B.clear(), f.#y());
  }
  /**
   * Traverse the effect tree, executing effects or stashing
   * them for later execution as appropriate
   * @param {Effect} root
   * @param {Effect[]} effects
   * @param {Effect[]} render_effects
   */
  #E(t, n, r) {
    t.f ^= E;
    for (var s = t.first; s !== null; ) {
      var i = s.f, a = (i & (L | Y)) !== 0, f = a && (i & E) !== 0, l = f || (i & O) !== 0 || this.#o.has(s);
      if (!l && s.fn !== null) {
        a ? s.f ^= E : (i & ce) !== 0 ? n.push(s) : ve(s) && ((i & M) !== 0 && this.#a.add(s), le(s));
        var u = s.first;
        if (u !== null) {
          s = u;
          continue;
        }
      }
      for (; s !== null; ) {
        var _ = s.next;
        if (_ !== null) {
          s = _;
          break;
        }
        s = s.parent;
      }
    }
  }
  #w() {
    for (var t = this.#v; t !== null; ) {
      if (!t.is_fork) {
        for (const [n, [, r]] of this.current)
          if (t.current.has(n) && !r)
            return t;
      }
      t = t.#v;
    }
    return null;
  }
  /**
   * @param {Batch} batch
   */
  #b(t) {
    for (const [r, s] of t.current)
      !this.previous.has(r) && t.previous.has(r) && this.previous.set(r, t.previous.get(r)), this.current.set(r, s);
    for (const [r, s] of t.async_deriveds) {
      const i = this.async_deriveds.get(r);
      i && s.promise.then(i.resolve).catch(i.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#f, t.#a);
    const n = (r) => {
      var s = r.reactions;
      if (s !== null && !((r.f & T) !== 0 && (r.f & (b | R)) === 0))
        for (const f of s) {
          var i = f.f;
          if ((i & T) !== 0)
            n(
              /** @type {Derived} */
              f
            );
          else {
            var a = (
              /** @type {Effect} */
              f
            );
            i & (oe | M) && !this.async_deriveds.has(a) && (this.#a.delete(a), w(a, b), this.schedule(a));
          }
        }
    };
    for (const r of this.current.keys())
      n(r);
    this.oncommit(() => t.discard()), t.#_(), y = this, this.#y();
  }
  /**
   * @param {Effect[]} effects
   */
  #h(t) {
    for (var n = 0; n < t.length; n += 1)
      xt(t[n], this.#f, this.#a);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== S && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & Z) === 0 && (this.current.set(t, [n, r]), N?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    y = this;
  }
  deactivate() {
    y = null, N = null;
  }
  flush() {
    try {
      Ue = !0, y = this, this.#y();
    } finally {
      ht = 0, We = null, ue = null, Oe = null, Ue = !1, y = null, N = null, B.clear();
    }
  }
  discard() {
    for (const t of this.#n) t(this);
    this.#n.clear();
    for (const t of this.async_deriveds.values())
      t.reject(pe);
    this.#_(), this.#s?.resolve();
  }
  /**
   * @param {Effect} effect
   */
  register_created_effect(t) {
    this.#d.push(t);
  }
  #m() {
    for (let o = He; o !== null; o = o.#r) {
      var t = o.id < this.id, n = [];
      for (const [c, [h, d]] of this.current) {
        if (o.current.has(c)) {
          var r = (
            /** @type {[any, boolean]} */
            o.current.get(c)[0]
          );
          if (t && h !== r)
            o.current.set(c, [h, d]);
          else
            continue;
        }
        n.push(c);
      }
      if (t)
        for (const [c, h] of this.async_deriveds) {
          const d = o.async_deriveds.get(c);
          d && h.promise.then(d.resolve).catch(d.reject);
        }
      var s = [...o.current.keys()].filter(
        (c) => !/** @type {[any, boolean]} */
        o.current.get(c)[1]
      );
      if (!(!o.#i || s.length === 0)) {
        var i = s.filter((c) => !this.current.has(c));
        if (i.length === 0)
          t && o.discard();
        else if (n.length > 0) {
          if (t)
            for (const c of this.#p)
              o.unskip_effect(c, (h) => {
                (h.f & (M | oe)) !== 0 ? o.schedule(h) : o.#h([h]);
              });
          o.activate();
          var a = /* @__PURE__ */ new Set(), f = /* @__PURE__ */ new Map();
          for (var l of n)
            Pt(l, i, a, f);
          f = /* @__PURE__ */ new Map();
          var u = [...o.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (u.length > 0)
            for (const c of this.#d)
              (c.f & (F | O | Ne)) === 0 && st(c, u, f) && ((c.f & (oe | M)) !== 0 ? (w(c, b), o.schedule(c)) : o.#f.add(c));
          if (o.#t.length > 0 && !o.#u) {
            o.apply();
            for (var _ of o.#t)
              o.#E(_, [], []);
            o.#t = [];
          }
          o.deactivate();
        }
      }
    }
  }
  /**
   * @param {boolean} blocking
   * @param {Effect} effect
   */
  increment(t, n) {
    if (this.#l += 1, t) {
      let r = this.#e.get(n) ?? 0;
      this.#e.set(n, r + 1);
    }
  }
  /**
   * @param {boolean} blocking
   * @param {Effect} effect
   */
  decrement(t, n) {
    if (this.#l -= 1, t) {
      let r = this.#e.get(n) ?? 0;
      r === 1 ? this.#e.delete(n) : this.#e.set(n, r - 1);
    }
    this.#u || (this.#u = !0, K(() => {
      this.#u = !1, this.linked && this.flush();
    }));
  }
  /**
   * @param {Set<Effect>} dirty_effects
   * @param {Set<Effect>} maybe_dirty_effects
   */
  transfer_effects(t, n) {
    for (const r of t)
      this.#f.add(r);
    for (const r of n)
      this.#a.add(r);
    t.clear(), n.clear();
  }
  /** @param {(batch: Batch) => void} fn */
  oncommit(t) {
    this.#c.add(t);
  }
  /** @param {(batch: Batch) => void} fn */
  ondiscard(t) {
    this.#n.add(t);
  }
  settled() {
    return (this.#s ??= mt()).promise;
  }
  static ensure() {
    if (y === null) {
      const t = y = new J();
      !Ue && !me && K(() => {
        t.#i || t.flush();
      });
    }
    return y;
  }
  apply() {
    {
      N = null;
      return;
    }
  }
  /**
   *
   * @param {Effect} effect
   */
  schedule(t) {
    if (We = t, t.b?.is_pending && (t.f & (ce | Te | et)) !== 0 && (t.f & _e) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (ue !== null && n === p && (v === null || (v.f & T) === 0))
        return;
      if ((r & (Y | L)) !== 0) {
        if ((r & E) === 0)
          return;
        n.f ^= E;
      }
    }
    this.#t.push(n);
  }
  #_() {
    if (this.linked) {
      var t = this.#v, n = this.#r;
      t === null ? He = n : t.#r = n, n === null ? ae = t : n.#v = t, this.linked = !1;
    }
  }
}
function Yn(e) {
  var t = me;
  me = !0;
  try {
    for (var n; ; ) {
      if (Pn(), y === null)
        return (
          /** @type {T} */
          n
        );
      y.flush();
    }
  } finally {
    me = t;
  }
}
function Gn() {
  try {
    mn();
  } catch (e) {
    W(e, We);
  }
}
let U = null;
function _t(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (F | O)) === 0 && ve(r) && (U = /* @__PURE__ */ new Set(), le(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Yt(r), U?.size > 0)) {
        B.clear();
        for (const s of U) {
          if ((s.f & (F | O)) !== 0) continue;
          const i = [s];
          let a = s.parent;
          for (; a !== null; )
            U.has(a) && (U.delete(a), i.push(a)), a = a.parent;
          for (let f = i.length - 1; f >= 0; f--) {
            const l = i[f];
            (l.f & (F | O)) === 0 && le(l);
          }
        }
        U.clear();
      }
    }
    U = null;
  }
}
function Pt(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const s of e.reactions) {
      const i = s.f;
      (i & T) !== 0 ? Pt(
        /** @type {Derived} */
        s,
        t,
        n,
        r
      ) : (i & (oe | M)) !== 0 && (i & b) === 0 && st(s, t, r) && (w(s, b), it(
        /** @type {Effect} */
        s
      ));
    }
}
function st(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const s of e.deps) {
      if (Me.call(t, s))
        return !0;
      if ((s.f & T) !== 0 && st(
        /** @type {Derived} */
        s,
        t,
        n
      ))
        return n.set(
          /** @type {Derived} */
          s,
          !0
        ), !0;
    }
  return n.set(e, !1), !1;
}
function it(e) {
  y.schedule(e);
}
function It(e, t) {
  if (!((e.f & L) !== 0 && (e.f & E) !== 0)) {
    (e.f & b) !== 0 ? t.d.push(e) : (e.f & R) !== 0 && t.m.push(e), w(e, E);
    for (var n = e.first; n !== null; )
      It(n, t), n = n.next;
  }
}
function Mt(e) {
  w(e, E);
  for (var t = e.first; t !== null; )
    Mt(t), t = t.next;
}
let Le = /* @__PURE__ */ new Set();
const B = /* @__PURE__ */ new Map();
let Nt = !1;
function Ce(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: bt,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function $(e, t) {
  const n = Ce(e);
  return zt(n), n;
}
// @__NO_SIDE_EFFECTS__
function Nr(e, t = !1, n = !0) {
  const r = Ce(e);
  return t || (r.equals = St), Be && n && m !== null && m.l !== null && (m.l.s ??= []).push(r), r;
}
function Dr(e, t) {
  return z(
    e,
    ut(() => X(e))
  ), t;
}
function z(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!D || (v.f & Ne) !== 0) && Ae() && (v.f & (T | M | oe | Ne)) !== 0 && (q === null || !q.has(e)) && Sn();
  let r = n ? ye(t) : t;
  return je(e, r, Oe);
}
function je(e, t, n = null) {
  if (!e.equals(t)) {
    Q ? B.set(e, t) : B.has(e) || B.set(e, e.v);
    var r = J.ensure();
    if (r.capture(e, t), (e.f & T) !== 0) {
      const s = (
        /** @type {Derived} */
        e
      );
      (e.f & b) !== 0 && rt(s), N === null && tt(s);
    }
    e.wv = Wt(), Dt(e, b, n), Ae() && p !== null && (p.f & E) !== 0 && (p.f & (L | Y)) === 0 && (x === null ? nr([e]) : x.push(e)), !r.is_fork && Le.size > 0 && !Nt && $n();
  }
  return t;
}
function $n() {
  Nt = !1;
  for (const e of Le) {
    (e.f & E) !== 0 && w(e, R);
    let t;
    try {
      t = ve(e);
    } catch {
      t = !0;
    }
    t && le(e);
  }
  Le.clear();
}
function Ee(e) {
  z(e, e.v + 1);
}
function Dt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var s = Ae(), i = r.length, a = 0; a < i; a++) {
      var f = r[a], l = f.f;
      if (!(!s && f === p)) {
        var u = (l & b) === 0;
        if (u && w(f, t), (l & Ne) !== 0)
          Le.add(
            /** @type {Effect} */
            f
          );
        else if ((l & T) !== 0) {
          var _ = (
            /** @type {Derived} */
            f
          );
          N?.delete(_), (l & ie) === 0 && (l & C && (p === null || (p.f & De) === 0) && (f.f |= ie), Dt(_, R, n));
        } else if (u) {
          var o = (
            /** @type {Effect} */
            f
          );
          (l & M) !== 0 && U !== null && U.add(o), n !== null ? n.push(o) : it(o);
        }
      }
    }
}
function ye(e) {
  if (typeof e != "object" || e === null || re in e)
    return e;
  const t = wt(e);
  if (t !== an && t !== un)
    return e;
  var n = /* @__PURE__ */ new Map(), r = nn(e), s = /* @__PURE__ */ $(0), i = se, a = (f) => {
    if (se === i)
      return f();
    var l = v, u = se;
    P(null), yt(i);
    var _ = f();
    return P(l), yt(u), _;
  };
  return r && n.set("length", /* @__PURE__ */ $(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(f, l, u) {
        (!("value" in u) || u.configurable === !1 || u.enumerable === !1 || u.writable === !1) && En();
        var _ = n.get(l);
        return _ === void 0 ? a(() => {
          var o = /* @__PURE__ */ $(u.value);
          return n.set(l, o), o;
        }) : z(_, u.value, !0), !0;
      },
      deleteProperty(f, l) {
        var u = n.get(l);
        if (u === void 0) {
          if (l in f) {
            const _ = a(() => /* @__PURE__ */ $(S));
            n.set(l, _), Ee(s);
          }
        } else
          z(u, S), Ee(s);
        return !0;
      },
      get(f, l, u) {
        if (l === re)
          return e;
        var _ = n.get(l), o = l in f;
        if (_ === void 0 && (!o || we(f, l)?.writable) && (_ = a(() => {
          var h = ye(o ? f[l] : S), d = /* @__PURE__ */ $(h);
          return d;
        }), n.set(l, _)), _ !== void 0) {
          var c = X(_);
          return c === S ? void 0 : c;
        }
        return Reflect.get(f, l, u);
      },
      getOwnPropertyDescriptor(f, l) {
        var u = Reflect.getOwnPropertyDescriptor(f, l);
        if (u && "value" in u) {
          var _ = n.get(l);
          _ && (u.value = X(_));
        } else if (u === void 0) {
          var o = n.get(l), c = o?.v;
          if (o !== void 0 && c !== S)
            return {
              enumerable: !0,
              configurable: !0,
              value: c,
              writable: !0
            };
        }
        return u;
      },
      has(f, l) {
        if (l === re)
          return !0;
        var u = n.get(l), _ = u !== void 0 && u.v !== S || Reflect.has(f, l);
        if (u !== void 0 || p !== null && (!_ || we(f, l)?.writable)) {
          u === void 0 && (u = a(() => {
            var c = _ ? ye(f[l]) : S, h = /* @__PURE__ */ $(c);
            return h;
          }), n.set(l, u));
          var o = X(u);
          if (o === S)
            return !1;
        }
        return _;
      },
      set(f, l, u, _) {
        var o = n.get(l), c = l in f;
        if (r && l === "length")
          for (var h = u; h < /** @type {Source<number>} */
          o.v; h += 1) {
            var d = n.get(h + "");
            d !== void 0 ? z(d, S) : h in f && (d = a(() => /* @__PURE__ */ $(S)), n.set(h + "", d));
          }
        if (o === void 0)
          (!c || we(f, l)?.writable) && (o = a(() => /* @__PURE__ */ $(void 0)), z(o, ye(u)), n.set(l, o));
        else {
          c = o.v !== S;
          var g = a(() => ye(u));
          z(o, g);
        }
        var G = Reflect.getOwnPropertyDescriptor(f, l);
        if (G?.set && G.set.call(_, u), !c) {
          if (r && typeof l == "string") {
            var H = (
              /** @type {Source<number>} */
              n.get("length")
            ), fe = Number(l);
            Number.isInteger(fe) && fe >= H.v && z(H, fe + 1);
          }
          Ee(s);
        }
        return !0;
      },
      ownKeys(f) {
        X(s);
        var l = Reflect.ownKeys(f).filter((o) => {
          var c = n.get(o);
          return c === void 0 || c.v !== S;
        });
        for (var [u, _] of n)
          _.v !== S && !(u in f) && l.push(u);
        return l;
      },
      setPrototypeOf() {
        bn();
      }
    }
  );
}
function dt(e) {
  try {
    if (e !== null && typeof e == "object" && re in e)
      return e[re];
  } catch {
  }
  return e;
}
function Fr(e, t) {
  return Object.is(dt(e), dt(t));
}
var vt, zn, Ft, Lt;
function Kn() {
  if (vt === void 0) {
    vt = window, zn = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    Ft = we(t, "firstChild").get, Lt = we(t, "nextSibling").get, ot(e) && (e[dn] = void 0, e[_n] = null, e[vn] = void 0, e.__e = void 0), ot(n) && (n[ze] = void 0);
  }
}
function jt(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function Bt(e) {
  return (
    /** @type {TemplateNode | null} */
    Ft.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function Ve(e) {
  return (
    /** @type {TemplateNode | null} */
    Lt.call(e)
  );
}
function Lr(e, t) {
  return /* @__PURE__ */ Bt(e);
}
function jr(e, t = !1) {
  {
    var n = /* @__PURE__ */ Bt(e);
    return n instanceof Comment && n.data === "" ? /* @__PURE__ */ Ve(n) : n;
  }
}
function Br(e, t = 1, n = !1) {
  let r = e;
  for (; t--; )
    r = /** @type {TemplateNode} */
    /* @__PURE__ */ Ve(r);
  return r;
}
function Vr(e) {
  e.textContent = "";
}
function qr() {
  return !1;
}
function Hr(e, t, n) {
  return t == null || t === tn ? (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  ) : (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElementNS(t, e, { is: n }) : document.createElementNS(t, e)
  );
}
function Vt(e) {
  p === null && (v === null && wn(), gn()), Q && yn();
}
function Wn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function j(e, t) {
  var n = p;
  n !== null && (n.f & O) !== 0 && (e |= O);
  var r = {
    ctx: m,
    deps: null,
    nodes: null,
    f: e | b | C,
    first: null,
    fn: t,
    last: null,
    next: null,
    parent: n,
    b: n && n.b,
    prev: null,
    teardown: null,
    wv: 0,
    ac: null
  };
  y?.register_created_effect(r);
  var s = r;
  if ((e & ce) !== 0)
    ue !== null ? ue.push(r) : J.ensure().schedule(r);
  else if (t !== null) {
    try {
      le(r);
    } catch (a) {
      throw V(r), a;
    }
    s.deps === null && s.teardown === null && s.nodes === null && s.first === s.last && // either `null`, or a singular child
    (s.f & de) === 0 && (s = s.first, (e & M) !== 0 && (e & be) !== 0 && s !== null && (s.f |= be));
  }
  if (s !== null && (s.parent = n, n !== null && Wn(s, n), v !== null && (v.f & T) !== 0 && (e & Y) === 0)) {
    var i = (
      /** @type {Derived} */
      v
    );
    (i.effects ??= []).push(s);
  }
  return r;
}
function lt() {
  return v !== null && !D;
}
function qt(e) {
  const t = j(Te, null);
  return w(t, E), t.teardown = e, t;
}
function Ur(e) {
  Vt();
  var t = (
    /** @type {Effect} */
    p.f
  ), n = !v && (t & L) !== 0 && m !== null && !m.i;
  if (n) {
    var r = (
      /** @type {ComponentContext} */
      m
    );
    (r.e ??= []).push(e);
  } else
    return Ht(e);
}
function Ht(e) {
  return j(ce | Et, e);
}
function Yr(e) {
  return Vt(), j(Te | Et, e);
}
function Xn(e) {
  J.ensure();
  const t = j(Y | de, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? Pe(t, () => {
      V(t), r(void 0);
    }) : (V(t), r(void 0));
  });
}
function Gr(e) {
  return j(ce, e);
}
function $r(e, t) {
  var n = (
    /** @type {ComponentContextLegacy} */
    m
  ), r = { effect: null, ran: !1, deps: e };
  n.l.$.push(r), r.effect = ft(() => {
    if (e(), !r.ran) {
      r.ran = !0;
      var s = (
        /** @type {Effect} */
        p
      );
      try {
        I(s.parent), ut(t);
      } finally {
        I(s);
      }
    }
  });
}
function zr() {
  var e = (
    /** @type {ComponentContextLegacy} */
    m
  );
  ft(() => {
    for (var t of e.l.$) {
      t.deps();
      var n = t.effect;
      (n.f & E) !== 0 && n.deps !== null && w(n, R), ve(n) && le(n), t.ran = !1;
    }
  });
}
function Zn(e) {
  return j(oe | de, e);
}
function ft(e, t = 0) {
  return j(Te | t, e);
}
function Kr(e, t = [], n = [], r = []) {
  Ln(r, t, n, (s) => {
    j(Te, () => {
      e(...s.map(X));
    });
  });
}
function Jn(e, t = 0) {
  var n = j(M | t, e);
  return n;
}
function Wr(e, t = 0) {
  var n = j(et | t, e);
  return n;
}
function ee(e) {
  return j(L | de, e);
}
function Ut(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = Q, r = v;
    pt(!0), P(null);
    try {
      t.call(null);
    } finally {
      pt(n), P(r);
    }
  }
}
function at(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const s = n.ac;
    s !== null && xe(() => {
      s.abort(ke);
    });
    var r = n.next;
    (n.f & Y) !== 0 ? n.parent = null : V(n, t), n = r;
  }
}
function Qn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & L) === 0 && V(t), t = n;
  }
}
function V(e, t = !0) {
  var n = !1;
  (t || (e.f & hn) !== 0) && e.nodes !== null && e.nodes.end !== null && (er(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= ct, at(e, t && !n), Se(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const i of r)
      i.stop();
  Ut(e), e.f ^= ct, e.f |= F;
  var s = e.parent;
  s !== null && s.first !== null && Yt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function er(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ Ve(e);
    e.remove(), e = n;
  }
}
function Yt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function Pe(e, t, n = !0) {
  var r = [];
  Gt(e, r, !0);
  var s = () => {
    n && V(e), t && t();
  }, i = r.length;
  if (i > 0) {
    var a = () => --i || s();
    for (var f of r)
      f.out(a);
  } else
    s();
}
function Gt(e, t, n) {
  if ((e.f & O) === 0) {
    e.f ^= O;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const f of r)
        (f.is_global || n) && t.push(f);
    for (var s = e.first; s !== null; ) {
      var i = s.next;
      if ((s.f & Y) === 0) {
        var a = (s.f & be) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (s.f & L) !== 0 && (e.f & M) !== 0;
        Gt(s, t, a ? n : !1);
      }
      s = i;
    }
  }
}
function Xr(e) {
  $t(e, !0);
}
function $t(e, t) {
  if ((e.f & O) !== 0) {
    e.f ^= O, (e.f & E) === 0 && (w(e, b), J.ensure().schedule(e));
    for (var n = e.first; n !== null; ) {
      var r = n.next, s = (n.f & be) !== 0 || (n.f & L) !== 0;
      $t(n, s ? t : !1), n = r;
    }
    var i = e.nodes && e.nodes.t;
    if (i !== null)
      for (const a of i)
        (a.is_global || t) && a.in();
  }
}
function tr(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var s = n === r ? null : /* @__PURE__ */ Ve(n);
      t.append(n), n = s;
    }
}
let Ie = !1, Q = !1;
function pt(e) {
  Q = e;
}
let v = null, D = !1;
function P(e) {
  v = e;
}
let p = null;
function I(e) {
  p = e;
}
let q = null;
function zt(e) {
  v !== null && (q ??= /* @__PURE__ */ new Set()).add(e);
}
let k = null, A = 0, x = null;
function nr(e) {
  x = e;
}
let Kt = 1, ne = 0, se = ne;
function yt(e) {
  se = e;
}
function Wt() {
  return ++Kt;
}
function ve(e) {
  var t = e.f;
  if ((t & b) !== 0)
    return !0;
  if (t & T && (e.f &= ~ie), (t & R) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, s = 0; s < r; s++) {
      var i = n[s];
      if (ve(
        /** @type {Derived} */
        i
      ) && Rt(
        /** @type {Derived} */
        i
      ), i.wv > e.wv)
        return !0;
    }
    (t & C) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    N === null && w(e, E);
  }
  return !1;
}
function Xt(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(q !== null && q.has(e)))
    for (var s = 0; s < r.length; s++) {
      var i = r[s];
      (i.f & T) !== 0 ? Xt(
        /** @type {Derived} */
        i,
        t,
        !1
      ) : t === i && (n ? w(i, b) : (i.f & E) !== 0 && w(i, R), it(
        /** @type {Effect} */
        i
      ));
    }
}
function Zt(e) {
  var t = k, n = A, r = x, s = v, i = q, a = m, f = D, l = se, u = e.f;
  k = /** @type {null | Value[]} */
  null, A = 0, x = null, v = (u & (L | Y)) === 0 ? e : null, q = null, he(e.ctx), D = !1, se = ++ne, e.ac !== null && (xe(() => {
    e.ac.abort(ke);
  }), e.ac = null);
  try {
    e.f |= De;
    var _ = (
      /** @type {Function} */
      e.fn
    ), o = _();
    e.f |= _e;
    var c = e.deps, h = y?.is_fork;
    if (k !== null) {
      var d;
      if (h || Se(e, A), c !== null && A > 0)
        for (c.length = A + k.length, d = 0; d < k.length; d++)
          c[A + d] = k[d];
      else
        e.deps = c = k;
      if (lt() && (e.f & C) !== 0)
        for (d = A; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && A < c.length && (Se(e, A), c.length = A);
    if (Ae() && x !== null && !D && c !== null && (e.f & (T | R | b)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      x.length; d++)
        Xt(
          x[d],
          /** @type {Effect} */
          e
        );
    if (s !== null && s !== e) {
      if (ne++, s.deps !== null)
        for (let g = 0; g < n; g += 1)
          s.deps[g].rv = ne;
      if (t !== null)
        for (const g of t)
          g.rv = ne;
      x !== null && (r === null ? r = x : r.push(.../** @type {Source[]} */
      x));
    }
    return (e.f & Z) !== 0 && (e.f ^= Z), o;
  } catch (g) {
    return kt(g);
  } finally {
    e.f ^= De, k = t, A = n, x = r, v = s, q = i, he(a), D = f, se = l;
  }
}
function rr(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = rn.call(n, e);
    if (r !== -1) {
      var s = n.length - 1;
      s === 0 ? n = t.reactions = null : (n[r] = n[s], n.pop());
    }
  }
  if (n === null && (t.f & T) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (k === null || !Me.call(k, t))) {
    var i = (
      /** @type {Derived} */
      t
    );
    (i.f & C) !== 0 && (i.f ^= C, i.f &= ~ie), i.v !== S && tt(i), i.ac !== null && xe(() => {
      i.ac.abort(ke), i.ac = null, w(i, b);
    }), Hn(i), Se(i, 0);
  }
}
function Se(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      rr(e, n[r]);
}
function le(e) {
  var t = e.f;
  if ((t & F) === 0) {
    w(e, E);
    var n = p, r = Ie;
    p = e, Ie = (t & (L | Y)) === 0;
    try {
      (t & (M | et)) !== 0 ? Qn(e) : at(e), Ut(e);
      var s = Zt(e);
      e.teardown = typeof s == "function" ? s : null, e.wv = Kt;
      var i;
      gt && Cn && (e.f & b) !== 0 && e.deps;
    } finally {
      Ie = r, p = n;
    }
  }
}
async function Zr() {
  await Promise.resolve(), Yn();
}
function X(e) {
  var t = e.f, n = (t & T) !== 0;
  if (v !== null && !D) {
    var r = p !== null && (p.f & F) !== 0;
    if (!r && (q === null || !q.has(e))) {
      var s = v.deps;
      if ((v.f & De) !== 0)
        e.rv < ne && (e.rv = ne, k === null && s !== null && s[A] === e ? A++ : k === null ? k = [e] : k.push(e));
      else {
        v.deps ??= [], Me.call(v.deps, e) || v.deps.push(e);
        var i = e.reactions;
        i === null ? e.reactions = [v] : Me.call(i, v) || i.push(v);
      }
    }
  }
  if (Q && B.has(e))
    return B.get(e);
  if (n) {
    var a = (
      /** @type {Derived} */
      e
    );
    if (Q) {
      var f = a.v;
      return ((a.f & E) === 0 && a.reactions !== null || Qt(a)) && (f = rt(a)), B.set(a, f), f;
    }
    var l = (a.f & C) === 0 && !D && v !== null && (Ie || (v.f & C) !== 0), u = (a.f & _e) === 0;
    ve(a) && (l && (a.f |= C), Rt(a)), l && !u && (Ot(a), Jt(a));
  }
  if (N?.has(e))
    return N.get(e);
  if ((e.f & Z) !== 0)
    throw e.v;
  return e.v;
}
function Jt(e) {
  if (e.f |= C, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & T) !== 0 && (t.f & C) === 0 && (Ot(
        /** @type {Derived} */
        t
      ), Jt(
        /** @type {Derived} */
        t
      ));
}
function Qt(e) {
  if (e.v === S) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (B.has(t) || (t.f & T) !== 0 && Qt(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function ut(e) {
  var t = D;
  try {
    return D = !0, e();
  } finally {
    D = t;
  }
}
function Jr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (re in e)
      Xe(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && re in n && Xe(n);
      }
  }
}
function Xe(e, t = /* @__PURE__ */ new Set()) {
  if (typeof e == "object" && e !== null && // We don't want to traverse DOM elements
  !(e instanceof EventTarget) && !t.has(e)) {
    t.add(e), e instanceof Date && e.getTime();
    for (let r in e)
      try {
        Xe(e[r], t);
      } catch {
      }
    const n = wt(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = fn(n);
      for (let s in r) {
        const i = r[s].get;
        if (i)
          try {
            i.call(e);
          } catch {
          }
      }
    }
  }
}
const ge = /* @__PURE__ */ Symbol("events"), en = /* @__PURE__ */ new Set(), Ze = /* @__PURE__ */ new Set();
function sr(e, t, n, r = {}) {
  function s(i) {
    if (r.capture || Je.call(t, i), !i.cancelBubble)
      return xe(() => n?.call(this, i));
  }
  return e.startsWith("pointer") || e.startsWith("touch") || e === "wheel" ? K(() => {
    t.addEventListener(e, s, r);
  }) : t.addEventListener(e, s, r), s;
}
function Qr(e, t, n, r, s) {
  var i = { capture: r, passive: s }, a = sr(e, t, n, i);
  (t === document.body || // @ts-ignore
  t === window || // @ts-ignore
  t === document || // Firefox has quirky behavior, it can happen that we still get "canplay" events when the element is already removed
  t instanceof HTMLMediaElement) && qt(() => {
    t.removeEventListener(e, a, i);
  });
}
function es(e, t, n) {
  (t[ge] ??= {})[e] = n;
}
function ts(e) {
  for (var t = 0; t < e.length; t++)
    en.add(e[t]);
  for (var n of Ze)
    n(e);
}
let Ye = null, Ge = !1;
function Je(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, s = e.composedPath?.() || [], i = (
    /** @type {null | Element} */
    s[0] || e.target
  );
  Ye = e, Ge || (Ge = !0, setTimeout(() => {
    Ge = !1, Ye = null;
  }));
  var a = 0, f = Ye === e && e[ge];
  if (f) {
    var l = s.indexOf(f);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[ge] = t;
      return;
    }
    var u = s.indexOf(t);
    if (u === -1)
      return;
    l <= u && (a = l);
  }
  if (i = /** @type {Element} */
  s[a] || e.target, i !== t) {
    ln(e, "currentTarget", {
      configurable: !0,
      get() {
        return i || n;
      }
    });
    var _ = v, o = p;
    P(null), I(null);
    try {
      for (var c, h = []; i !== null && i !== t; ) {
        try {
          var d = i[ge]?.[r];
          d != null && (!/** @type {any} */
          i.disabled || // DOM could've been updated already by the time this is reached, so we check this as well
          // -> the target could not have been disabled because it emits the event in the first place
          e.target === i) && d.call(i, e);
        } catch (g) {
          c ? h.push(g) : c = g;
        }
        if (e.cancelBubble) break;
        a++, i = a < s.length ? (
          /** @type {Element} */
          s[a]
        ) : null;
      }
      if (c) {
        for (let g of h)
          queueMicrotask(() => {
            throw g;
          });
        throw c;
      }
    } finally {
      e[ge] = t, delete e.currentTarget, P(_), I(o);
    }
  }
}
function ns(e) {
  return e.endsWith("capture") && e !== "gotpointercapture" && e !== "lostpointercapture";
}
const ir = [
  "beforeinput",
  "click",
  "change",
  "dblclick",
  "contextmenu",
  "focusin",
  "focusout",
  "input",
  "keydown",
  "keyup",
  "mousedown",
  "mousemove",
  "mouseout",
  "mouseover",
  "mouseup",
  "pointerdown",
  "pointermove",
  "pointerout",
  "pointerover",
  "pointerup",
  "touchend",
  "touchmove",
  "touchstart"
];
function rs(e) {
  return ir.includes(e);
}
const lr = {
  // no `class: 'className'` because we handle that separately
  formnovalidate: "formNoValidate",
  ismap: "isMap",
  nomodule: "noModule",
  playsinline: "playsInline",
  readonly: "readOnly",
  defaultvalue: "defaultValue",
  defaultchecked: "defaultChecked",
  srcobject: "srcObject",
  novalidate: "noValidate",
  allowfullscreen: "allowFullscreen",
  disablepictureinpicture: "disablePictureInPicture",
  disableremoteplayback: "disableRemotePlayback"
};
function ss(e) {
  return e = e.toLowerCase(), lr[e] ?? e;
}
const fr = ["touchstart", "touchmove"];
function ar(e) {
  return fr.includes(e);
}
function is(e, t) {
  var n = t == null ? "" : typeof t == "object" ? `${t}` : t;
  n !== /** @type {any} */
  (e[ze] ??= e.nodeValue) && (e[ze] = n, e.nodeValue = `${n}`);
}
function ls(e, t) {
  return ur(e, t);
}
const Re = /* @__PURE__ */ new Map();
function ur(e, { target: t, anchor: n, props: r = {}, events: s, context: i, intro: a = !0, transformError: f }) {
  Kn();
  var l = void 0, u = Xn(() => {
    var _ = n ?? t.appendChild(jt());
    Dn(
      /** @type {TemplateNode} */
      _,
      {
        pending: () => {
        }
      },
      (h) => {
        Rn({});
        var d = (
          /** @type {ComponentContext} */
          m
        );
        i && (d.c = i), s && (r.$$events = s), l = e(h, r) || {}, On();
      },
      f
    );
    var o = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var g = h[d];
        if (!o.has(g)) {
          o.add(g);
          var G = ar(g);
          for (const qe of [t, document]) {
            var H = Re.get(qe);
            H === void 0 && (H = /* @__PURE__ */ new Map(), Re.set(qe, H));
            var fe = H.get(g);
            fe === void 0 ? (qe.addEventListener(g, Je, { passive: G }), H.set(g, 1)) : H.set(g, fe + 1);
          }
        }
      }
    };
    return c(sn(en)), Ze.add(c), () => {
      for (var h of o)
        for (const G of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            Re.get(G)
          ), g = (
            /** @type {number} */
            d.get(h)
          );
          --g == 0 ? (G.removeEventListener(h, Je), d.delete(h), d.size === 0 && Re.delete(G)) : d.set(h, g);
        }
      Ze.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return Qe.set(l, u), l;
}
let Qe = /* @__PURE__ */ new WeakMap();
function fs(e, t) {
  const n = Qe.get(e);
  return n ? (Qe.delete(e), n(t)) : Promise.resolve();
}
export {
  Ar as $,
  Jn as A,
  m as B,
  Be as C,
  Ur as D,
  be as E,
  je as F,
  Rr as G,
  Vn as H,
  sn as I,
  cr as J,
  Ce as K,
  or as L,
  hr as M,
  Tr as N,
  F as O,
  O as P,
  L as Q,
  Vr as R,
  Ve as S,
  gr as T,
  mr as U,
  Wr as V,
  Gr as W,
  dn as X,
  vn as Y,
  Pr as Z,
  Fr as _,
  z as a,
  _n as a0,
  tn as a1,
  fn as a2,
  Ln as a3,
  Er as a4,
  Cr as a5,
  ns as a6,
  es as a7,
  ts as a8,
  sr as a9,
  On as aA,
  Rn as aB,
  Br as aC,
  Qr as aD,
  Kr as aE,
  Dr as aF,
  Lr as aG,
  is as aH,
  Mr as aI,
  Zr as aJ,
  ls as aK,
  fs as aL,
  ss as aa,
  S as ab,
  rs as ac,
  ft as ad,
  ct as ae,
  re as af,
  Yr as ag,
  Sr as ah,
  Jr as ai,
  nt as aj,
  we as ak,
  Or as al,
  vr as am,
  ye as an,
  pr as ao,
  dr as ap,
  _r as aq,
  yr as ar,
  Q as as,
  kr as at,
  br as au,
  $ as av,
  Ir as aw,
  $r as ax,
  zr as ay,
  jr as az,
  X as b,
  Hr as c,
  ln as d,
  jt as e,
  p as f,
  wt as g,
  Bt as h,
  nn as i,
  zn as j,
  wr as k,
  Xr as l,
  Nr as m,
  on as n,
  an as o,
  V as p,
  K as q,
  cn as r,
  xn as s,
  qt as t,
  ut as u,
  Pe as v,
  ee as w,
  y as x,
  tr as y,
  qr as z
};
