const ur = 1, or = 2, cr = 4, hr = 8, _r = 16, dr = 1, vr = 2, pr = 4, yr = 8, gr = 16, wr = 1, Er = 2, b = /* @__PURE__ */ Symbol("uninitialized"), en = "http://www.w3.org/1999/xhtml", mr = "http://www.w3.org/2000/svg", br = "@attach", yt = !1;
var tn = Array.isArray, nn = Array.prototype.indexOf, Pe = Array.prototype.includes, rn = Array.from, sn = Object.defineProperty, we = Object.getOwnPropertyDescriptor, ln = Object.getOwnPropertyDescriptors, fn = Object.prototype, an = Array.prototype, gt = Object.getPrototypeOf, at = Object.isExtensible;
function Sr(e) {
  return typeof e == "function";
}
const un = () => {
};
function Tr(e) {
  return e();
}
function on(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function wt() {
  var e, t, n = new Promise((r, s) => {
    e = r, t = s;
  });
  return { promise: n, resolve: e, reject: t };
}
const T = 2, ce = 4, Te = 8, Ze = 1 << 24, I = 16, H = 32, W = 64, Ue = 128, C = 512, m = 1024, S = 2048, F = 4096, R = 8192, D = 16384, _e = 32768, ut = 1 << 25, be = 65536, Ie = 1 << 17, cn = 1 << 18, de = 1 << 19, Et = 1 << 20, Ar = 1 << 25, ie = 65536, Ne = 1 << 21, oe = 1 << 22, K = 1 << 23, te = /* @__PURE__ */ Symbol("$state"), kr = /* @__PURE__ */ Symbol("legacy props"), xr = /* @__PURE__ */ Symbol(""), hn = /* @__PURE__ */ Symbol("attributes"), _n = /* @__PURE__ */ Symbol("class"), dn = /* @__PURE__ */ Symbol("style"), Ye = /* @__PURE__ */ Symbol("text"), Le = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}(), Rr = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  !!globalThis.document?.contentType && /* @__PURE__ */ globalThis.document.contentType.includes("xml")
);
function vn() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function Or(e, t, n) {
  throw new Error("https://svelte.dev/e/each_key_duplicate");
}
function pn(e) {
  throw new Error("https://svelte.dev/e/effect_in_teardown");
}
function yn() {
  throw new Error("https://svelte.dev/e/effect_in_unowned_derived");
}
function gn(e) {
  throw new Error("https://svelte.dev/e/effect_orphan");
}
function wn() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function Pr(e) {
  throw new Error("https://svelte.dev/e/props_invalid_value");
}
function En() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function mn() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function bn() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function Sn() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function Tn() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function Ir() {
  console.warn("https://svelte.dev/e/select_multiple_invalid_value");
}
function An() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function mt(e) {
  return e === this.v;
}
function kn(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function bt(e) {
  return !kn(e, this.v);
}
let je = !1, xn = !1;
function Nr() {
  je = !0;
}
let E = null;
function he(e) {
  E = e;
}
function Cn(e, t = !1, n) {
  E = {
    p: E,
    i: !1,
    c: null,
    e: null,
    s: e,
    x: null,
    r: (
      /** @type {Effect} */
      p
    ),
    l: je && !t ? { s: null, u: null, $: [] } : null
  };
}
function Rn(e) {
  var t = (
    /** @type {ComponentContext} */
    E
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      Vt(r);
  }
  return t.i = !0, E = t.p, /** @type {T} */
  {};
}
function Ae() {
  return !je || E !== null && E.l === null;
}
let Q = [];
function St() {
  var e = Q;
  Q = [], on(e);
}
function ne(e) {
  if (Q.length === 0 && !Ee) {
    var t = Q;
    queueMicrotask(() => {
      t === Q && St();
    });
  }
  Q.push(e);
}
function On() {
  for (; Q.length > 0; )
    St();
}
function Tt(e) {
  var t = p;
  if (t === null)
    return v.f |= K, e;
  if ((t.f & _e) === 0 && (t.f & ce) === 0)
    throw e;
  $(e, t);
}
function $(e, t) {
  if (!(t !== null && (t.f & D) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & Ue) !== 0) {
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
const Pn = -7169;
function w(e, t) {
  e.f = e.f & Pn | t;
}
function Je(e) {
  (e.f & C) !== 0 || e.deps === null ? w(e, m) : w(e, F);
}
function At(e) {
  if (e !== null)
    for (const t of e)
      (t.f & T) === 0 || (t.f & ie) === 0 || (t.f ^= ie, At(
        /** @type {Derived} */
        t.deps
      ));
}
function kt(e, t, n) {
  (e.f & S) !== 0 ? t.add(e) : (e.f & F) !== 0 && n.add(e), At(e.deps), w(e, m);
}
function In(e) {
  let t = 0, n = ke(0), r;
  return () => {
    st() && (z(n), it(() => (t === 0 && (r = ft(() => e(() => me(n)))), t += 1, () => {
      ne(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, me(n));
      });
    })));
  };
}
var Nn = be | de;
function Mn(e, t, n, r) {
  new Dn(e, t, n, r);
}
class Dn {
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
  #h;
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
  #_ = 0;
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
  #g = In(() => (this.#u = ke(this.#_), () => {
    this.#u = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(t, n, r, s) {
    this.#i = t, this.#r = n, this.#h = (i) => {
      var u = (
        /** @type {Effect} */
        p
      );
      u.b = this, u.f |= Ue, r(i);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = s ?? this.parent?.transform_error ?? ((i) => i), this.#n = Zn(() => {
      this.#w();
    }, Nn);
  }
  #y() {
    try {
      this.#l = J(() => this.#h(this.#i));
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #m(t) {
    const n = this.#r.failed;
    n && (this.#s = J(() => {
      n(
        this.#i,
        () => t,
        () => () => {
        }
      );
    }));
  }
  #b() {
    const t = this.#r.pending;
    t && (this.is_pending = !0, this.#e = J(() => t(this.#i)), ne(() => {
      var n = this.#t = document.createDocumentFragment(), r = Lt();
      n.append(r), this.#l = this.#E(() => J(() => this.#h(r))), this.#f === 0 && (this.#i.before(n), this.#t = null, Re(
        /** @type {Effect} */
        this.#e,
        () => {
          this.#e = null;
        }
      ), this.#c(
        /** @type {Batch} */
        y
      ));
    }));
  }
  #w() {
    try {
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#_ = 0, this.#l = J(() => {
        this.#h(this.#i);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        er(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = J(() => n(this.#i));
      } else
        this.#c(
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
  #c(t) {
    this.is_pending = !1, t.transfer_effects(this.#o, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(t) {
    kt(t, this.#o, this.#p);
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
  #E(t) {
    var n = p, r = v, s = E;
    P(this.#n), O(this.#n), he(this.#n.ctx);
    try {
      return X.ensure(), t();
    } catch (i) {
      return Tt(i), null;
    } finally {
      P(n), O(r), he(s);
    }
  }
  /**
   * Updates the pending count associated with the currently visible pending snippet,
   * if any, such that we can replace the snippet with content once work is done
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  #d(t, n) {
    if (!this.has_pending_snippet()) {
      this.parent && this.parent.#d(t, n);
      return;
    }
    this.#f += t, this.#f === 0 && (this.#c(n), this.#e && Re(this.#e, () => {
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
    this.#d(t, n), this.#_ += t, !(!this.#u || this.#a) && (this.#a = !0, ne(() => {
      this.#a = !1, this.#u && Fe(this.#u, this.#_);
    }));
  }
  get_effect_pending() {
    return this.#g(), z(
      /** @type {Source<number>} */
      this.#u
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    y?.is_fork ? (this.#l && y.skip_effect(this.#l), this.#e && y.skip_effect(this.#e), this.#s && y.skip_effect(this.#s), y.oncommit(() => {
      this.#S(t);
    })) : this.#S(t);
  }
  /**
   * @param {unknown} error
   */
  #S(t) {
    this.#l && (j(this.#l), this.#l = null), this.#e && (j(this.#e), this.#e = null), this.#s && (j(this.#s), this.#s = null);
    var n = this.#r.onerror;
    let r = this.#r.failed;
    var s = !1, i = !1;
    const u = () => {
      if (s) {
        An();
        return;
      }
      s = !0, i && Sn(), this.#s !== null && Re(this.#s, () => {
        this.#s = null;
      }), this.#E(() => {
        this.#w();
      });
    }, a = (l) => {
      try {
        i = !0, n?.(l, u), i = !1;
      } catch (f) {
        $(f, this.#n && this.#n.parent);
      }
      r && (this.#s = this.#E(() => {
        try {
          return J(() => {
            var f = (
              /** @type {Effect} */
              p
            );
            f.b = this, f.f |= Ue, r(
              this.#i,
              () => l,
              () => u
            );
          });
        } catch (f) {
          return $(
            f,
            /** @type {Effect} */
            this.#n.parent
          ), null;
        }
      }));
    };
    ne(() => {
      var l;
      try {
        l = this.transform_error(t);
      } catch (f) {
        $(f, this.#n && this.#n.parent);
        return;
      }
      l !== null && typeof l == "object" && typeof /** @type {any} */
      l.then == "function" ? l.then(
        a,
        /** @param {unknown} e */
        (f) => $(f, this.#n && this.#n.parent)
      ) : a(l);
    });
  }
}
function Fn(e, t, n, r) {
  const s = Ae() ? Qe : Bn;
  var i = e.filter((h) => !h.settled), u = t.map(s);
  if (n.length === 0 && i.length === 0) {
    r(u);
    return;
  }
  var a = (
    /** @type {Effect} */
    p
  ), l = Ln(), f = i.length === 1 ? i[0].promise : i.length > 1 ? Promise.all(i.map((h) => h.promise)) : null;
  function _(h) {
    if ((a.f & D) === 0) {
      l();
      try {
        r([...u, ...h]);
      } catch (d) {
        $(d, a);
      }
      Me();
    }
  }
  var o = xt();
  if (n.length === 0) {
    f.then(() => _([])).finally(o);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ jn(h))).then(_).catch((h) => $(h, a)).finally(o);
  }
  f ? f.then(() => {
    l(), c(), Me();
  }) : c();
}
function Ln() {
  var e = (
    /** @type {Effect} */
    p
  ), t = v, n = E, r = (
    /** @type {Batch} */
    y
  );
  return function(i = !0) {
    P(e), O(t), he(n), i && (e.f & D) === 0 && (r?.activate(), r?.apply());
  };
}
function Me(e = !0) {
  P(null), O(null), he(null), e && y?.deactivate();
}
function xt() {
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
function Qe(e) {
  var t = T | S;
  return p !== null && (p.f |= de), {
    ctx: E,
    deps: null,
    effects: null,
    equals: mt,
    f: t,
    fn: e,
    reactions: null,
    rv: 0,
    v: (
      /** @type {V} */
      b
    ),
    wv: 0,
    parent: p,
    ac: null
  };
}
const pe = /* @__PURE__ */ Symbol("obsolete");
// @__NO_SIDE_EFFECTS__
function jn(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && vn();
  var s = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), i = ke(
    /** @type {V} */
    b
  ), u = !v, a = /* @__PURE__ */ new Set();
  return Xn(() => {
    var l = (
      /** @type {Effect} */
      p
    ), f = wt();
    s = f.promise;
    try {
      Promise.resolve(e()).then(f.resolve, (h) => {
        h !== Le && f.reject(h);
      }).finally(Me);
    } catch (h) {
      f.reject(h), Me();
    }
    var _ = (
      /** @type {Batch} */
      y
    );
    if (u) {
      if ((l.f & _e) !== 0)
        var o = xt();
      if (
        // boundary can be null if the async derived is inside an $effect.root not connected to the component render tree
        r.b?.is_rendered()
      )
        _.async_deriveds.get(l)?.reject(pe);
      else
        for (const h of a.values())
          h.reject(pe);
      a.add(f), _.async_deriveds.set(l, f);
    }
    const c = (h, d = void 0) => {
      o?.(), a.delete(f), d !== pe && (_.activate(), d ? (i.f |= K, Fe(i, d)) : ((i.f & K) !== 0 && (i.f ^= K), Fe(i, h)), _.deactivate());
    };
    f.promise.then(c, (h) => c(null, h || "unknown"));
  }), Ht(() => {
    for (const l of a)
      l.reject(pe);
  }), new Promise((l) => {
    function f(_) {
      function o() {
        _ === s ? l(i) : f(s);
      }
      _.then(o, o);
    }
    f(s);
  });
}
// @__NO_SIDE_EFFECTS__
function Mr(e) {
  const t = /* @__PURE__ */ Qe(e);
  return $t(t), t;
}
// @__NO_SIDE_EFFECTS__
function Bn(e) {
  const t = /* @__PURE__ */ Qe(e);
  return t.equals = bt, t;
}
function Hn(e) {
  var t = e.effects;
  if (t !== null) {
    e.effects = null;
    for (var n = 0; n < t.length; n += 1)
      j(
        /** @type {Effect} */
        t[n]
      );
  }
}
function et(e) {
  var t, n = p, r = e.parent;
  if (!Z && r !== null && e.v !== b && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (D | R)) !== 0)
    return Tn(), e.v;
  P(r);
  try {
    e.f &= ~ie, Hn(e), t = Xt(e);
  } finally {
    P(n);
  }
  return t;
}
function Ct(e) {
  var t = et(e);
  if (!e.equals(t) && (e.wv = Kt(), (!y?.is_fork || e.deps === null) && (y !== null ? (y.capture(e, t, !0), Ge?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, m);
    return;
  }
  Z || (N !== null ? (st() || y?.is_fork) && N.set(e, t) : Je(e));
}
function Vn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac?.abort(Le), t.fn !== null && (t.teardown = un), t.ac = null, Se(t, 0), lt(t));
}
function Rt(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && le(t);
}
let Ve = null, ae = null, y = null, Ge = null, N = null, $e = null, Ee = !1, qe = !1, ue = null, Ce = null;
var ot = 0;
let qn = 1;
class X {
  id = qn++;
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
  #h = /* @__PURE__ */ new Set();
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
  #_ = [];
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
    ae === null ? Ve = ae = this : (ae.#r = this, this.#v = ae), ae = this;
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
        w(s, S), n(s);
      for (s of r.m)
        w(s, F), n(s);
    }
    this.#p.add(t);
  }
  #y() {
    this.#i = !0, ot++ > 1e3 && (this.#d(), Yn());
    for (const l of this.#f)
      this.#a.delete(l), w(l, S), this.schedule(l);
    for (const l of this.#a)
      w(l, F), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = ue = [], r = [], s = Ce = [];
    for (const l of t)
      try {
        this.#m(l, n, r);
      } catch (f) {
        throw It(l), this.#g() || this.discard(), f;
      }
    if (y = null, s.length > 0) {
      var i = X.ensure();
      for (const l of s)
        i.schedule(l);
    }
    if (ue = null, Ce = null, this.#g()) {
      this.#c(r), this.#c(n);
      for (const [l, f] of this.#o)
        Pt(l, f);
      s.length > 0 && /** @type {unknown} */
      y.#y();
      return;
    }
    const u = this.#b();
    if (u) {
      this.#c(r), this.#c(n), u.#w(this);
      return;
    }
    this.#f.clear(), this.#a.clear();
    for (const l of this.#h) l(this);
    this.#h.clear(), Ge = this, ct(r), ct(n), Ge = null, this.#s?.resolve();
    var a = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      y
    );
    if (this.#l === 0 && (this.#t.length === 0 || a !== null) && this.#d(), this.#t.length > 0)
      if (a !== null) {
        const l = a;
        l.#t.push(...this.#t.filter((f) => !l.#t.includes(f)));
      } else
        a = this;
    a !== null && a.#y();
  }
  /**
   * Traverse the effect tree, executing effects or stashing
   * them for later execution as appropriate
   * @param {Effect} root
   * @param {Effect[]} effects
   * @param {Effect[]} render_effects
   */
  #m(t, n, r) {
    t.f ^= m;
    for (var s = t.first; s !== null; ) {
      var i = s.f, u = (i & (H | W)) !== 0, a = u && (i & m) !== 0, l = a || (i & R) !== 0 || this.#o.has(s);
      if (!l && s.fn !== null) {
        u ? s.f ^= m : (i & ce) !== 0 ? n.push(s) : ve(s) && ((i & I) !== 0 && this.#a.add(s), le(s));
        var f = s.first;
        if (f !== null) {
          s = f;
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
  #b() {
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
  #w(t) {
    for (const [r, s] of t.current)
      !this.previous.has(r) && t.previous.has(r) && this.previous.set(r, t.previous.get(r)), this.current.set(r, s);
    for (const [r, s] of t.async_deriveds) {
      const i = this.async_deriveds.get(r);
      i && s.promise.then(i.resolve).catch(i.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#f, t.#a);
    const n = (r) => {
      var s = r.reactions;
      if (s !== null)
        for (const a of s) {
          var i = a.f;
          if ((i & T) !== 0)
            n(
              /** @type {Derived} */
              a
            );
          else {
            var u = (
              /** @type {Effect} */
              a
            );
            i & (oe | I) && !this.async_deriveds.has(u) && (this.#a.delete(u), w(u, S), this.schedule(u));
          }
        }
    };
    for (const r of this.current.keys())
      n(r);
    this.oncommit(() => t.discard()), t.#d(), y = this, this.#y();
  }
  /**
   * @param {Effect[]} effects
   */
  #c(t) {
    for (var n = 0; n < t.length; n += 1)
      kt(t[n], this.#f, this.#a);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== b && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & K) === 0 && (this.current.set(t, [n, r]), N?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    y = this;
  }
  deactivate() {
    y = null, N = null;
  }
  flush() {
    try {
      qe = !0, y = this, this.#y();
    } finally {
      ot = 0, $e = null, ue = null, Ce = null, qe = !1, y = null, N = null, re.clear();
    }
  }
  discard() {
    for (const t of this.#n) t(this);
    this.#n.clear();
    for (const t of this.async_deriveds.values())
      t.reject(pe);
    this.#d(), this.#s?.resolve();
  }
  /**
   * @param {Effect} effect
   */
  register_created_effect(t) {
    this.#_.push(t);
  }
  #E() {
    for (let o = Ve; o !== null; o = o.#r) {
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
                (h.f & (I | oe)) !== 0 ? o.schedule(h) : o.#c([h]);
              });
          o.activate();
          var u = /* @__PURE__ */ new Set(), a = /* @__PURE__ */ new Map();
          for (var l of n)
            Ot(l, i, u, a);
          a = /* @__PURE__ */ new Map();
          var f = [...o.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (f.length > 0)
            for (const c of this.#_)
              (c.f & (D | R | Ie)) === 0 && tt(c, f, a) && ((c.f & (oe | I)) !== 0 ? (w(c, S), o.schedule(c)) : o.#f.add(c));
          if (o.#t.length > 0 && !o.#u) {
            o.apply();
            for (var _ of o.#t)
              o.#m(_, [], []);
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
    this.#u || (this.#u = !0, ne(() => {
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
    this.#h.add(t);
  }
  /** @param {(batch: Batch) => void} fn */
  ondiscard(t) {
    this.#n.add(t);
  }
  settled() {
    return (this.#s ??= wt()).promise;
  }
  static ensure() {
    if (y === null) {
      const t = y = new X();
      !qe && !Ee && ne(() => {
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
    if ($e = t, t.b?.is_pending && (t.f & (ce | Te | Ze)) !== 0 && (t.f & _e) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (ue !== null && n === p && (v === null || (v.f & T) === 0))
        return;
      if ((r & (W | H)) !== 0) {
        if ((r & m) === 0)
          return;
        n.f ^= m;
      }
    }
    this.#t.push(n);
  }
  #d() {
    if (this.linked) {
      var t = this.#v, n = this.#r;
      t === null ? Ve = n : t.#r = n, n === null ? ae = t : n.#v = t, this.linked = !1;
    }
  }
}
function Un(e) {
  var t = Ee;
  Ee = !0;
  try {
    for (var n; ; ) {
      if (On(), y === null)
        return (
          /** @type {T} */
          n
        );
      y.flush();
    }
  } finally {
    Ee = t;
  }
}
function Yn() {
  try {
    wn();
  } catch (e) {
    $(e, $e);
  }
}
let q = null;
function ct(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (D | R)) === 0 && ve(r) && (q = /* @__PURE__ */ new Set(), le(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Ut(r), q?.size > 0)) {
        re.clear();
        for (const s of q) {
          if ((s.f & (D | R)) !== 0) continue;
          const i = [s];
          let u = s.parent;
          for (; u !== null; )
            q.has(u) && (q.delete(u), i.push(u)), u = u.parent;
          for (let a = i.length - 1; a >= 0; a--) {
            const l = i[a];
            (l.f & (D | R)) === 0 && le(l);
          }
        }
        q.clear();
      }
    }
    q = null;
  }
}
function Ot(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const s of e.reactions) {
      const i = s.f;
      (i & T) !== 0 ? Ot(
        /** @type {Derived} */
        s,
        t,
        n,
        r
      ) : (i & (oe | I)) !== 0 && (i & S) === 0 && tt(s, t, r) && (w(s, S), nt(
        /** @type {Effect} */
        s
      ));
    }
}
function tt(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const s of e.deps) {
      if (Pe.call(t, s))
        return !0;
      if ((s.f & T) !== 0 && tt(
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
function nt(e) {
  y.schedule(e);
}
function Pt(e, t) {
  if (!((e.f & H) !== 0 && (e.f & m) !== 0)) {
    (e.f & S) !== 0 ? t.d.push(e) : (e.f & F) !== 0 && t.m.push(e), w(e, m);
    for (var n = e.first; n !== null; )
      Pt(n, t), n = n.next;
  }
}
function It(e) {
  w(e, m);
  for (var t = e.first; t !== null; )
    It(t), t = t.next;
}
let De = /* @__PURE__ */ new Set();
const re = /* @__PURE__ */ new Map();
let Nt = !1;
function ke(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: mt,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function Y(e, t) {
  const n = ke(e);
  return $t(n), n;
}
// @__NO_SIDE_EFFECTS__
function Dr(e, t = !1, n = !0) {
  const r = ke(e);
  return t || (r.equals = bt), je && n && E !== null && E.l !== null && (E.l.s ??= []).push(r), r;
}
function Fr(e, t) {
  return G(
    e,
    ft(() => z(e))
  ), t;
}
function G(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!M || (v.f & Ie) !== 0) && Ae() && (v.f & (T | I | oe | Ie)) !== 0 && (B === null || !B.has(e)) && bn();
  let r = n ? ye(t) : t;
  return Fe(e, r, Ce);
}
function Fe(e, t, n = null) {
  if (!e.equals(t)) {
    re.set(e, Z ? t : e.v);
    var r = X.ensure();
    if (r.capture(e, t), (e.f & T) !== 0) {
      const s = (
        /** @type {Derived} */
        e
      );
      (e.f & S) !== 0 && et(s), N === null && Je(s);
    }
    e.wv = Kt(), Mt(e, S, n), Ae() && p !== null && (p.f & m) !== 0 && (p.f & (H | W)) === 0 && (x === null ? tr([e]) : x.push(e)), !r.is_fork && De.size > 0 && !Nt && Gn();
  }
  return t;
}
function Gn() {
  Nt = !1;
  for (const e of De) {
    (e.f & m) !== 0 && w(e, F);
    let t;
    try {
      t = ve(e);
    } catch {
      t = !0;
    }
    t && le(e);
  }
  De.clear();
}
function me(e) {
  G(e, e.v + 1);
}
function Mt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var s = Ae(), i = r.length, u = 0; u < i; u++) {
      var a = r[u], l = a.f;
      if (!(!s && a === p)) {
        var f = (l & S) === 0;
        if (f && w(a, t), (l & Ie) !== 0)
          De.add(
            /** @type {Effect} */
            a
          );
        else if ((l & T) !== 0) {
          var _ = (
            /** @type {Derived} */
            a
          );
          N?.delete(_), (l & ie) === 0 && (l & C && (p === null || (p.f & Ne) === 0) && (a.f |= ie), Mt(_, F, n));
        } else if (f) {
          var o = (
            /** @type {Effect} */
            a
          );
          (l & I) !== 0 && q !== null && q.add(o), n !== null ? n.push(o) : nt(o);
        }
      }
    }
}
function ye(e) {
  if (typeof e != "object" || e === null || te in e)
    return e;
  const t = gt(e);
  if (t !== fn && t !== an)
    return e;
  var n = /* @__PURE__ */ new Map(), r = tn(e), s = /* @__PURE__ */ Y(0), i = se, u = (a) => {
    if (se === i)
      return a();
    var l = v, f = se;
    O(null), vt(i);
    var _ = a();
    return O(l), vt(f), _;
  };
  return r && n.set("length", /* @__PURE__ */ Y(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(a, l, f) {
        (!("value" in f) || f.configurable === !1 || f.enumerable === !1 || f.writable === !1) && En();
        var _ = n.get(l);
        return _ === void 0 ? u(() => {
          var o = /* @__PURE__ */ Y(f.value);
          return n.set(l, o), o;
        }) : G(_, f.value, !0), !0;
      },
      deleteProperty(a, l) {
        var f = n.get(l);
        if (f === void 0) {
          if (l in a) {
            const _ = u(() => /* @__PURE__ */ Y(b));
            n.set(l, _), me(s);
          }
        } else
          G(f, b), me(s);
        return !0;
      },
      get(a, l, f) {
        if (l === te)
          return e;
        var _ = n.get(l), o = l in a;
        if (_ === void 0 && (!o || we(a, l)?.writable) && (_ = u(() => {
          var h = ye(o ? a[l] : b), d = /* @__PURE__ */ Y(h);
          return d;
        }), n.set(l, _)), _ !== void 0) {
          var c = z(_);
          return c === b ? void 0 : c;
        }
        return Reflect.get(a, l, f);
      },
      getOwnPropertyDescriptor(a, l) {
        var f = Reflect.getOwnPropertyDescriptor(a, l);
        if (f && "value" in f) {
          var _ = n.get(l);
          _ && (f.value = z(_));
        } else if (f === void 0) {
          var o = n.get(l), c = o?.v;
          if (o !== void 0 && c !== b)
            return {
              enumerable: !0,
              configurable: !0,
              value: c,
              writable: !0
            };
        }
        return f;
      },
      has(a, l) {
        if (l === te)
          return !0;
        var f = n.get(l), _ = f !== void 0 && f.v !== b || Reflect.has(a, l);
        if (f !== void 0 || p !== null && (!_ || we(a, l)?.writable)) {
          f === void 0 && (f = u(() => {
            var c = _ ? ye(a[l]) : b, h = /* @__PURE__ */ Y(c);
            return h;
          }), n.set(l, f));
          var o = z(f);
          if (o === b)
            return !1;
        }
        return _;
      },
      set(a, l, f, _) {
        var o = n.get(l), c = l in a;
        if (r && l === "length")
          for (var h = f; h < /** @type {Source<number>} */
          o.v; h += 1) {
            var d = n.get(h + "");
            d !== void 0 ? G(d, b) : h in a && (d = u(() => /* @__PURE__ */ Y(b)), n.set(h + "", d));
          }
        if (o === void 0)
          (!c || we(a, l)?.writable) && (o = u(() => /* @__PURE__ */ Y(void 0)), G(o, ye(f)), n.set(l, o));
        else {
          c = o.v !== b;
          var g = u(() => ye(f));
          G(o, g);
        }
        var U = Reflect.getOwnPropertyDescriptor(a, l);
        if (U?.set && U.set.call(_, f), !c) {
          if (r && typeof l == "string") {
            var V = (
              /** @type {Source<number>} */
              n.get("length")
            ), fe = Number(l);
            Number.isInteger(fe) && fe >= V.v && G(V, fe + 1);
          }
          me(s);
        }
        return !0;
      },
      ownKeys(a) {
        z(s);
        var l = Reflect.ownKeys(a).filter((o) => {
          var c = n.get(o);
          return c === void 0 || c.v !== b;
        });
        for (var [f, _] of n)
          _.v !== b && !(f in a) && l.push(f);
        return l;
      },
      setPrototypeOf() {
        mn();
      }
    }
  );
}
function ht(e) {
  try {
    if (e !== null && typeof e == "object" && te in e)
      return e[te];
  } catch {
  }
  return e;
}
function Lr(e, t) {
  return Object.is(ht(e), ht(t));
}
var _t, $n, Dt, Ft;
function zn() {
  if (_t === void 0) {
    _t = window, $n = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    Dt = we(t, "firstChild").get, Ft = we(t, "nextSibling").get, at(e) && (e[_n] = void 0, e[hn] = null, e[dn] = void 0, e.__e = void 0), at(n) && (n[Ye] = void 0);
  }
}
function Lt(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function jt(e) {
  return (
    /** @type {TemplateNode | null} */
    Dt.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function Be(e) {
  return (
    /** @type {TemplateNode | null} */
    Ft.call(e)
  );
}
function jr(e, t) {
  return /* @__PURE__ */ jt(e);
}
function Br(e, t = !1) {
  {
    var n = /* @__PURE__ */ jt(e);
    return n instanceof Comment && n.data === "" ? /* @__PURE__ */ Be(n) : n;
  }
}
function Hr(e, t = 1, n = !1) {
  let r = e;
  for (; t--; )
    r = /** @type {TemplateNode} */
    /* @__PURE__ */ Be(r);
  return r;
}
function Vr(e) {
  e.textContent = "";
}
function qr() {
  return !1;
}
function Ur(e, t, n) {
  return t == null || t === en ? (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  ) : (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElementNS(t, e, { is: n }) : document.createElementNS(t, e)
  );
}
function rt(e) {
  var t = v, n = p;
  O(null), P(null);
  try {
    return e();
  } finally {
    O(t), P(n);
  }
}
function Bt(e) {
  p === null && (v === null && gn(), yn()), Z && pn();
}
function Kn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function L(e, t) {
  var n = p;
  n !== null && (n.f & R) !== 0 && (e |= R);
  var r = {
    ctx: E,
    deps: null,
    nodes: null,
    f: e | S | C,
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
    ue !== null ? ue.push(r) : X.ensure().schedule(r);
  else if (t !== null) {
    try {
      le(r);
    } catch (u) {
      throw j(r), u;
    }
    s.deps === null && s.teardown === null && s.nodes === null && s.first === s.last && // either `null`, or a singular child
    (s.f & de) === 0 && (s = s.first, (e & I) !== 0 && (e & be) !== 0 && s !== null && (s.f |= be));
  }
  if (s !== null && (s.parent = n, n !== null && Kn(s, n), v !== null && (v.f & T) !== 0 && (e & W) === 0)) {
    var i = (
      /** @type {Derived} */
      v
    );
    (i.effects ??= []).push(s);
  }
  return r;
}
function st() {
  return v !== null && !M;
}
function Ht(e) {
  const t = L(Te, null);
  return w(t, m), t.teardown = e, t;
}
function Yr(e) {
  Bt();
  var t = (
    /** @type {Effect} */
    p.f
  ), n = !v && (t & H) !== 0 && E !== null && !E.i;
  if (n) {
    var r = (
      /** @type {ComponentContext} */
      E
    );
    (r.e ??= []).push(e);
  } else
    return Vt(e);
}
function Vt(e) {
  return L(ce | Et, e);
}
function Gr(e) {
  return Bt(), L(Te | Et, e);
}
function Wn(e) {
  X.ensure();
  const t = L(W | de, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? Re(t, () => {
      j(t), r(void 0);
    }) : (j(t), r(void 0));
  });
}
function $r(e) {
  return L(ce, e);
}
function zr(e, t) {
  var n = (
    /** @type {ComponentContextLegacy} */
    E
  ), r = { effect: null, ran: !1, deps: e };
  n.l.$.push(r), r.effect = it(() => {
    if (e(), !r.ran) {
      r.ran = !0;
      var s = (
        /** @type {Effect} */
        p
      );
      try {
        P(s.parent), ft(t);
      } finally {
        P(s);
      }
    }
  });
}
function Kr() {
  var e = (
    /** @type {ComponentContextLegacy} */
    E
  );
  it(() => {
    for (var t of e.l.$) {
      t.deps();
      var n = t.effect;
      (n.f & m) !== 0 && n.deps !== null && w(n, F), ve(n) && le(n), t.ran = !1;
    }
  });
}
function Xn(e) {
  return L(oe | de, e);
}
function it(e, t = 0) {
  return L(Te | t, e);
}
function Wr(e, t = [], n = [], r = []) {
  Fn(r, t, n, (s) => {
    L(Te, () => {
      e(...s.map(z));
    });
  });
}
function Zn(e, t = 0) {
  var n = L(I | t, e);
  return n;
}
function Xr(e, t = 0) {
  var n = L(Ze | t, e);
  return n;
}
function J(e) {
  return L(H | de, e);
}
function qt(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = Z, r = v;
    dt(!0), O(null);
    try {
      t.call(null);
    } finally {
      dt(n), O(r);
    }
  }
}
function lt(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const s = n.ac;
    s !== null && rt(() => {
      s.abort(Le);
    });
    var r = n.next;
    (n.f & W) !== 0 ? n.parent = null : j(n, t), n = r;
  }
}
function Jn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & H) === 0 && j(t), t = n;
  }
}
function j(e, t = !0) {
  var n = !1;
  (t || (e.f & cn) !== 0) && e.nodes !== null && e.nodes.end !== null && (Qn(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= ut, lt(e, t && !n), Se(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const i of r)
      i.stop();
  qt(e), e.f ^= ut, e.f |= D;
  var s = e.parent;
  s !== null && s.first !== null && Ut(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function Qn(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ Be(e);
    e.remove(), e = n;
  }
}
function Ut(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function Re(e, t, n = !0) {
  var r = [];
  Yt(e, r, !0);
  var s = () => {
    n && j(e), t && t();
  }, i = r.length;
  if (i > 0) {
    var u = () => --i || s();
    for (var a of r)
      a.out(u);
  } else
    s();
}
function Yt(e, t, n) {
  if ((e.f & R) === 0) {
    e.f ^= R;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const a of r)
        (a.is_global || n) && t.push(a);
    for (var s = e.first; s !== null; ) {
      var i = s.next;
      if ((s.f & W) === 0) {
        var u = (s.f & be) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (s.f & H) !== 0 && (e.f & I) !== 0;
        Yt(s, t, u ? n : !1);
      }
      s = i;
    }
  }
}
function Zr(e) {
  Gt(e, !0);
}
function Gt(e, t) {
  if ((e.f & R) !== 0) {
    e.f ^= R, (e.f & m) === 0 && (w(e, S), X.ensure().schedule(e));
    for (var n = e.first; n !== null; ) {
      var r = n.next, s = (n.f & be) !== 0 || (n.f & H) !== 0;
      Gt(n, s ? t : !1), n = r;
    }
    var i = e.nodes && e.nodes.t;
    if (i !== null)
      for (const u of i)
        (u.is_global || t) && u.in();
  }
}
function er(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var s = n === r ? null : /* @__PURE__ */ Be(n);
      t.append(n), n = s;
    }
}
let Oe = !1, Z = !1;
function dt(e) {
  Z = e;
}
let v = null, M = !1;
function O(e) {
  v = e;
}
let p = null;
function P(e) {
  p = e;
}
let B = null;
function $t(e) {
  v !== null && (B ??= /* @__PURE__ */ new Set()).add(e);
}
let A = null, k = 0, x = null;
function tr(e) {
  x = e;
}
let zt = 1, ee = 0, se = ee;
function vt(e) {
  se = e;
}
function Kt() {
  return ++zt;
}
function ve(e) {
  var t = e.f;
  if ((t & S) !== 0)
    return !0;
  if (t & T && (e.f &= ~ie), (t & F) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, s = 0; s < r; s++) {
      var i = n[s];
      if (ve(
        /** @type {Derived} */
        i
      ) && Ct(
        /** @type {Derived} */
        i
      ), i.wv > e.wv)
        return !0;
    }
    (t & C) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    N === null && w(e, m);
  }
  return !1;
}
function Wt(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(B !== null && B.has(e)))
    for (var s = 0; s < r.length; s++) {
      var i = r[s];
      (i.f & T) !== 0 ? Wt(
        /** @type {Derived} */
        i,
        t,
        !1
      ) : t === i && (n ? w(i, S) : (i.f & m) !== 0 && w(i, F), nt(
        /** @type {Effect} */
        i
      ));
    }
}
function Xt(e) {
  var t = A, n = k, r = x, s = v, i = B, u = E, a = M, l = se, f = e.f;
  A = /** @type {null | Value[]} */
  null, k = 0, x = null, v = (f & (H | W)) === 0 ? e : null, B = null, he(e.ctx), M = !1, se = ++ee, e.ac !== null && (rt(() => {
    e.ac.abort(Le);
  }), e.ac = null);
  try {
    e.f |= Ne;
    var _ = (
      /** @type {Function} */
      e.fn
    ), o = _();
    e.f |= _e;
    var c = e.deps, h = y?.is_fork;
    if (A !== null) {
      var d;
      if (h || Se(e, k), c !== null && k > 0)
        for (c.length = k + A.length, d = 0; d < A.length; d++)
          c[k + d] = A[d];
      else
        e.deps = c = A;
      if (st() && (e.f & C) !== 0)
        for (d = k; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && k < c.length && (Se(e, k), c.length = k);
    if (Ae() && x !== null && !M && c !== null && (e.f & (T | F | S)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      x.length; d++)
        Wt(
          x[d],
          /** @type {Effect} */
          e
        );
    if (s !== null && s !== e) {
      if (ee++, s.deps !== null)
        for (let g = 0; g < n; g += 1)
          s.deps[g].rv = ee;
      if (t !== null)
        for (const g of t)
          g.rv = ee;
      x !== null && (r === null ? r = x : r.push(.../** @type {Source[]} */
      x));
    }
    return (e.f & K) !== 0 && (e.f ^= K), o;
  } catch (g) {
    return Tt(g);
  } finally {
    e.f ^= Ne, A = t, k = n, x = r, v = s, B = i, he(u), M = a, se = l;
  }
}
function nr(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = nn.call(n, e);
    if (r !== -1) {
      var s = n.length - 1;
      s === 0 ? n = t.reactions = null : (n[r] = n[s], n.pop());
    }
  }
  if (n === null && (t.f & T) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (A === null || !Pe.call(A, t))) {
    var i = (
      /** @type {Derived} */
      t
    );
    (i.f & C) !== 0 && (i.f ^= C, i.f &= ~ie), i.v !== b && Je(i), Vn(i), Se(i, 0);
  }
}
function Se(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      nr(e, n[r]);
}
function le(e) {
  var t = e.f;
  if ((t & D) === 0) {
    w(e, m);
    var n = p, r = Oe;
    p = e, Oe = !0;
    try {
      (t & (I | Ze)) !== 0 ? Jn(e) : lt(e), qt(e);
      var s = Xt(e);
      e.teardown = typeof s == "function" ? s : null, e.wv = zt;
      var i;
      yt && xn && (e.f & S) !== 0 && e.deps;
    } finally {
      Oe = r, p = n;
    }
  }
}
async function Jr() {
  await Promise.resolve(), Un();
}
function z(e) {
  var t = e.f, n = (t & T) !== 0;
  if (v !== null && !M) {
    var r = p !== null && (p.f & D) !== 0;
    if (!r && (B === null || !B.has(e))) {
      var s = v.deps;
      if ((v.f & Ne) !== 0)
        e.rv < ee && (e.rv = ee, A === null && s !== null && s[k] === e ? k++ : A === null ? A = [e] : A.push(e));
      else {
        v.deps ??= [], Pe.call(v.deps, e) || v.deps.push(e);
        var i = e.reactions;
        i === null ? e.reactions = [v] : Pe.call(i, v) || i.push(v);
      }
    }
  }
  if (Z && re.has(e))
    return re.get(e);
  if (n) {
    var u = (
      /** @type {Derived} */
      e
    );
    if (Z) {
      var a = u.v;
      return ((u.f & m) === 0 && u.reactions !== null || Jt(u)) && (a = et(u)), re.set(u, a), a;
    }
    var l = (u.f & C) === 0 && !M && v !== null && (Oe || (v.f & C) !== 0), f = (u.f & _e) === 0;
    ve(u) && (l && (u.f |= C), Ct(u)), l && !f && (Rt(u), Zt(u));
  }
  if (N?.has(e))
    return N.get(e);
  if ((e.f & K) !== 0)
    throw e.v;
  return e.v;
}
function Zt(e) {
  if (e.f |= C, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & T) !== 0 && (t.f & C) === 0 && (Rt(
        /** @type {Derived} */
        t
      ), Zt(
        /** @type {Derived} */
        t
      ));
}
function Jt(e) {
  if (e.v === b) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (re.has(t) || (t.f & T) !== 0 && Jt(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function ft(e) {
  var t = M;
  try {
    return M = !0, e();
  } finally {
    M = t;
  }
}
function Qr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (te in e)
      ze(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && te in n && ze(n);
      }
  }
}
function ze(e, t = /* @__PURE__ */ new Set()) {
  if (typeof e == "object" && e !== null && // We don't want to traverse DOM elements
  !(e instanceof EventTarget) && !t.has(e)) {
    t.add(e), e instanceof Date && e.getTime();
    for (let r in e)
      try {
        ze(e[r], t);
      } catch {
      }
    const n = gt(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = ln(n);
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
const ge = /* @__PURE__ */ Symbol("events"), Qt = /* @__PURE__ */ new Set(), Ke = /* @__PURE__ */ new Set();
function rr(e, t, n, r = {}) {
  function s(i) {
    if (r.capture || We.call(t, i), !i.cancelBubble)
      return rt(() => n?.call(this, i));
  }
  return e.startsWith("pointer") || e.startsWith("touch") || e === "wheel" ? ne(() => {
    t.addEventListener(e, s, r);
  }) : t.addEventListener(e, s, r), s;
}
function es(e, t, n, r, s) {
  var i = { capture: r, passive: s }, u = rr(e, t, n, i);
  (t === document.body || // @ts-ignore
  t === window || // @ts-ignore
  t === document || // Firefox has quirky behavior, it can happen that we still get "canplay" events when the element is already removed
  t instanceof HTMLMediaElement) && Ht(() => {
    t.removeEventListener(e, u, i);
  });
}
function ts(e, t, n) {
  (t[ge] ??= {})[e] = n;
}
function ns(e) {
  for (var t = 0; t < e.length; t++)
    Qt.add(e[t]);
  for (var n of Ke)
    n(e);
}
let pt = null;
function We(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, s = e.composedPath?.() || [], i = (
    /** @type {null | Element} */
    s[0] || e.target
  );
  pt = e;
  var u = 0, a = pt === e && e[ge];
  if (a) {
    var l = s.indexOf(a);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[ge] = t;
      return;
    }
    var f = s.indexOf(t);
    if (f === -1)
      return;
    l <= f && (u = l);
  }
  if (i = /** @type {Element} */
  s[u] || e.target, i !== t) {
    sn(e, "currentTarget", {
      configurable: !0,
      get() {
        return i || n;
      }
    });
    var _ = v, o = p;
    O(null), P(null);
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
        u++, i = u < s.length ? (
          /** @type {Element} */
          s[u]
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
      e[ge] = t, delete e.currentTarget, O(_), P(o);
    }
  }
}
function rs(e) {
  return e.endsWith("capture") && e !== "gotpointercapture" && e !== "lostpointercapture";
}
const sr = [
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
function ss(e) {
  return sr.includes(e);
}
const ir = {
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
function is(e) {
  return e = e.toLowerCase(), ir[e] ?? e;
}
const lr = ["touchstart", "touchmove"];
function fr(e) {
  return lr.includes(e);
}
function ls(e, t) {
  var n = t == null ? "" : typeof t == "object" ? `${t}` : t;
  n !== /** @type {any} */
  (e[Ye] ??= e.nodeValue) && (e[Ye] = n, e.nodeValue = `${n}`);
}
function fs(e, t) {
  return ar(e, t);
}
const xe = /* @__PURE__ */ new Map();
function ar(e, { target: t, anchor: n, props: r = {}, events: s, context: i, intro: u = !0, transformError: a }) {
  zn();
  var l = void 0, f = Wn(() => {
    var _ = n ?? t.appendChild(Lt());
    Mn(
      /** @type {TemplateNode} */
      _,
      {
        pending: () => {
        }
      },
      (h) => {
        Cn({});
        var d = (
          /** @type {ComponentContext} */
          E
        );
        i && (d.c = i), s && (r.$$events = s), l = e(h, r) || {}, Rn();
      },
      a
    );
    var o = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var g = h[d];
        if (!o.has(g)) {
          o.add(g);
          var U = fr(g);
          for (const He of [t, document]) {
            var V = xe.get(He);
            V === void 0 && (V = /* @__PURE__ */ new Map(), xe.set(He, V));
            var fe = V.get(g);
            fe === void 0 ? (He.addEventListener(g, We, { passive: U }), V.set(g, 1)) : V.set(g, fe + 1);
          }
        }
      }
    };
    return c(rn(Qt)), Ke.add(c), () => {
      for (var h of o)
        for (const U of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            xe.get(U)
          ), g = (
            /** @type {number} */
            d.get(h)
          );
          --g == 0 ? (U.removeEventListener(h, We), d.delete(h), d.size === 0 && xe.delete(U)) : d.set(h, g);
        }
      Ke.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return Xe.set(l, f), l;
}
let Xe = /* @__PURE__ */ new WeakMap();
function as(e, t) {
  const n = Xe.get(e);
  return n ? (Xe.delete(e), n(t)) : Promise.resolve();
}
export {
  Ir as $,
  Zn as A,
  E as B,
  je as C,
  Yr as D,
  be as E,
  Fe as F,
  Or as G,
  Bn as H,
  rn as I,
  or as J,
  ke as K,
  ur as L,
  _r as M,
  Ar as N,
  D as O,
  R as P,
  H as Q,
  Vr as R,
  Be as S,
  wr as T,
  cr as U,
  hr as V,
  mr as W,
  Xr as X,
  $r as Y,
  _n as Z,
  dn as _,
  G as a,
  Lr as a0,
  xr as a1,
  hn as a2,
  en as a3,
  ln as a4,
  Fn as a5,
  br as a6,
  Rr as a7,
  rs as a8,
  ts as a9,
  Kr as aA,
  Br as aB,
  Rn as aC,
  Cn as aD,
  Hr as aE,
  es as aF,
  Wr as aG,
  Fr as aH,
  jr as aI,
  ls as aJ,
  Mr as aK,
  Jr as aL,
  fs as aM,
  as as aN,
  ns as aa,
  rr as ab,
  is as ac,
  b as ad,
  ss as ae,
  it as af,
  ut as ag,
  te as ah,
  Gr as ai,
  Tr as aj,
  Qr as ak,
  Qe as al,
  we as am,
  Pr as an,
  pr as ao,
  ye as ap,
  yr as aq,
  vr as ar,
  dr as as,
  gr as at,
  Z as au,
  kr as av,
  Sr as aw,
  Y as ax,
  Nr as ay,
  zr as az,
  z as b,
  Ur as c,
  sn as d,
  Lt as e,
  p as f,
  gt as g,
  jt as h,
  tn as i,
  $n as j,
  Er as k,
  Zr as l,
  Dr as m,
  un as n,
  fn as o,
  j as p,
  ne as q,
  on as r,
  kn as s,
  Ht as t,
  ft as u,
  Re as v,
  J as w,
  y as x,
  er as y,
  qr as z
};
