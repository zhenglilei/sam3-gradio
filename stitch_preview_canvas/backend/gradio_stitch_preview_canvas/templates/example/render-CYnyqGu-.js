let We = !1, $t = !1;
function nr() {
  We = !0;
}
const rr = 2, sr = 4, ir = 8, lr = 2, b = /* @__PURE__ */ Symbol("uninitialized"), ht = !1;
var Gt = Array.isArray, Kt = Array.prototype.indexOf, Ce = Array.prototype.includes, Wt = Array.from, Jt = Object.defineProperty, ve = Object.getOwnPropertyDescriptor, Qt = Object.getOwnPropertyDescriptors, Xt = Object.prototype, Zt = Array.prototype, _t = Object.getPrototypeOf, rt = Object.isExtensible;
const en = () => {
};
function fr(e) {
  return e();
}
function tn(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function dt() {
  var e, t, n = new Promise((r, s) => {
    e = r, t = s;
  });
  return { promise: n, resolve: e, reject: t };
}
const S = 2, ye = 4, be = 8, vt = 1 << 24, P = 16, I = 32, V = 64, Ve = 128, R = 512, E = 1024, m = 2048, F = 4096, O = 8192, j = 16384, ce = 32768, st = 1 << 25, we = 65536, Pe = 1 << 17, nn = 1 << 18, he = 1 << 19, pt = 1 << 20, re = 65536, De = 1 << 21, ue = 1 << 22, K = 1 << 23, pe = /* @__PURE__ */ Symbol("$state"), ur = /* @__PURE__ */ Symbol("legacy props"), rn = /* @__PURE__ */ Symbol("attributes"), sn = /* @__PURE__ */ Symbol("class"), ln = /* @__PURE__ */ Symbol("style"), ze = /* @__PURE__ */ Symbol("text"), Ee = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}();
function fn() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function un(e) {
  throw new Error("https://svelte.dev/e/effect_in_teardown");
}
function on() {
  throw new Error("https://svelte.dev/e/effect_in_unowned_derived");
}
function an(e) {
  throw new Error("https://svelte.dev/e/effect_orphan");
}
function cn() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function ar(e) {
  throw new Error("https://svelte.dev/e/props_invalid_value");
}
function hn() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function _n() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function dn() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function vn() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function pn() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function gn() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function gt(e) {
  return e === this.v;
}
function yn(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function wn(e) {
  return !yn(e, this.v);
}
let k = null;
function oe(e) {
  k = e;
}
function mn(e, t = !1, n) {
  k = {
    p: k,
    i: !1,
    c: null,
    e: null,
    s: e,
    x: null,
    r: (
      /** @type {Effect} */
      p
    ),
    l: We && !t ? { s: null, u: null, $: [] } : null
  };
}
function bn(e) {
  var t = (
    /** @type {ComponentContext} */
    k
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      jt(r);
  }
  return t.i = !0, k = t.p, /** @type {T} */
  {};
}
function Se() {
  return !We || k !== null && k.l === null;
}
let le = [];
function En() {
  var e = le;
  le = [], tn(e);
}
function Z(e) {
  if (le.length === 0) {
    var t = le;
    queueMicrotask(() => {
      t === le && En();
    });
  }
  le.push(e);
}
function yt(e) {
  var t = p;
  if (t === null)
    return v.f |= K, e;
  if ((t.f & ce) === 0 && (t.f & ye) === 0)
    throw e;
  G(e, t);
}
function G(e, t) {
  if (!(t !== null && (t.f & j) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & Ve) !== 0) {
        if ((t.f & ce) === 0)
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
const Sn = -7169;
function w(e, t) {
  e.f = e.f & Sn | t;
}
function Je(e) {
  (e.f & R) !== 0 || e.deps === null ? w(e, E) : w(e, F);
}
function wt(e) {
  if (e !== null)
    for (const t of e)
      (t.f & S) === 0 || (t.f & re) === 0 || (t.f ^= re, wt(
        /** @type {Derived} */
        t.deps
      ));
}
function mt(e, t, n) {
  (e.f & m) !== 0 ? t.add(e) : (e.f & F) !== 0 && n.add(e), wt(e.deps), w(e, E);
}
function Ie(e) {
  var t = v, n = p;
  C(null), B(null);
  try {
    return e();
  } finally {
    C(t), B(n);
  }
}
function kn(e) {
  let t = 0, n = Me(0), r;
  return () => {
    tt() && (te(n), Vn(() => (t === 0 && (r = Qn(() => e(() => ge(n)))), t += 1, () => {
      Z(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, ge(n));
      });
    })));
  };
}
var xn = we | he;
function Tn(e, t, n, r) {
  new An(e, t, n, r);
}
class An {
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
  #u = !1;
  /** @type {Set<Effect>} */
  #a = /* @__PURE__ */ new Set();
  /** @type {Set<Effect>} */
  #p = /* @__PURE__ */ new Set();
  /**
   * A source containing the number of pending async deriveds/expressions.
   * Only created if `$effect.pending()` is used inside the boundary,
   * otherwise updating the source results in needless `Batch.ensure()`
   * calls followed by no-op flushes
   * @type {Source<number> | null}
   */
  #o = null;
  #y = kn(() => (this.#o = Me(this.#d), () => {
    this.#o = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(t, n, r, s) {
    this.#i = t, this.#r = n, this.#c = (i) => {
      var u = (
        /** @type {Effect} */
        p
      );
      u.b = this, u.f |= Ve, r(i);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = s ?? this.parent?.transform_error ?? ((i) => i), this.#n = zn(() => {
      this.#h();
    }, xn);
  }
  #g() {
    try {
      this.#l = Q(() => this.#c(this.#i));
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #b(t) {
    const n = this.#r.failed, { reset: r, invoke_onerror: s } = this.#w(t);
    Z(s), n && (this.#s = Q(() => {
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
        gn();
        return;
      }
      n = !0, r && vn(), this.#s !== null && Re(this.#s, () => {
        this.#s = null;
      }), this.#_(() => {
        this.#h();
      });
    };
    return { reset: s, invoke_onerror: () => {
      try {
        r = !0, this.#r.onerror?.(t, s), r = !1;
      } catch (u) {
        G(u, this.#n && this.#n.parent);
      }
    } };
  }
  #E() {
    const t = this.#r.pending;
    t && (this.is_pending = !0, this.#e = Q(() => t(this.#i)), Z(() => {
      var n = this.#t = document.createDocumentFragment(), r = Dt();
      n.append(r), this.#l = this.#_(() => Q(() => this.#c(r))), this.#f === 0 && (this.#i.before(n), this.#t = null, Re(
        /** @type {Effect} */
        this.#e,
        () => {
          this.#e = null;
        }
      ), this.#m(
        /** @type {Batch} */
        g
      ));
    }));
  }
  #h() {
    try {
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#d = 0, this.#l = Q(() => {
        this.#c(this.#i);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        Gn(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = Q(() => n(this.#i));
      } else
        this.#m(
          /** @type {Batch} */
          g
        );
    } catch (n) {
      this.error(n);
    }
  }
  /**
   * @param {Batch} batch
   */
  #m(t) {
    this.is_pending = !1, t.transfer_effects(this.#a, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(t) {
    mt(t, this.#a, this.#p);
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
    var n = p, r = v, s = k;
    B(this.#n), C(this.#n), oe(this.#n.ctx);
    try {
      return W.ensure(), t();
    } catch (i) {
      return yt(i), null;
    } finally {
      B(n), C(r), oe(s);
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
    this.#f += t, this.#f === 0 && (this.#m(n), this.#e && Re(this.#e, () => {
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
    this.#S(t, n), this.#d += t, !(!this.#o || this.#u) && (this.#u = !0, Z(() => {
      this.#u = !1, this.#o && je(this.#o, this.#d);
    }));
  }
  get_effect_pending() {
    return this.#y(), te(
      /** @type {Source<number>} */
      this.#o
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    g?.is_fork ? (this.#l && g.skip_effect(this.#l), this.#e && g.skip_effect(this.#e), this.#s && g.skip_effect(this.#s), g.oncommit(() => {
      this.#k(t);
    })) : this.#k(t);
  }
  /**
   * @param {unknown} error
   */
  #k(t) {
    this.#l && (L(this.#l), this.#l = null), this.#e && (L(this.#e), this.#e = null), this.#s && (L(this.#s), this.#s = null);
    let n = this.#r.failed;
    const r = (s) => {
      const { reset: i, invoke_onerror: u } = this.#w(s);
      u(), n && (this.#s = this.#_(() => {
        try {
          return Q(() => {
            var f = (
              /** @type {Effect} */
              p
            );
            f.b = this, f.f |= Ve, n(
              this.#i,
              () => s,
              () => i
            );
          });
        } catch (f) {
          return G(
            f,
            /** @type {Effect} */
            this.#n.parent
          ), null;
        }
      }));
    };
    Z(() => {
      var s;
      try {
        s = this.transform_error(t);
      } catch (i) {
        G(i, this.#n && this.#n.parent);
        return;
      }
      s !== null && typeof s == "object" && typeof /** @type {any} */
      s.then == "function" ? s.then(
        r,
        /** @param {unknown} e */
        (i) => G(i, this.#n && this.#n.parent)
      ) : r(s);
    });
  }
}
function Rn(e, t, n, r) {
  const s = Se() ? Et : Pn;
  var i = e.filter((h) => !h.settled), u = t.map(s);
  if (n.length === 0 && i.length === 0) {
    r(u);
    return;
  }
  var f = (
    /** @type {Effect} */
    p
  ), l = On(), o = i.length === 1 ? i[0].promise : i.length > 1 ? Promise.all(i.map((h) => h.promise)) : null;
  function _(h) {
    if ((f.f & j) === 0) {
      l();
      try {
        r([...u, ...h]);
      } catch (d) {
        G(d, f);
      }
      Ne();
    }
  }
  var a = bt();
  if (n.length === 0) {
    o.then(() => _([])).finally(a);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ Cn(h))).then(_).catch((h) => G(h, f)).finally(a);
  }
  o ? o.then(() => {
    l(), c(), Ne();
  }) : c();
}
function On() {
  var e = (
    /** @type {Effect} */
    p
  ), t = v, n = k, r = (
    /** @type {Batch} */
    g
  );
  return function(i = !0) {
    B(e), C(t), oe(n), i && (e.f & j) === 0 && (r?.activate(), r?.apply());
  };
}
function Ne(e = !0) {
  B(null), C(null), oe(null), e && g?.deactivate();
}
function bt() {
  var e = (
    /** @type {Effect} */
    p
  ), t = e.b, n = (
    /** @type {Batch} */
    g
  ), r = !!t?.is_rendered();
  return t?.update_pending_count(1, n), n.increment(r, e), () => {
    t?.update_pending_count(-1, n), n.decrement(r, e);
  };
}
// @__NO_SIDE_EFFECTS__
function Et(e) {
  var t = S | m;
  return p !== null && (p.f |= he), {
    ctx: k,
    deps: null,
    effects: null,
    equals: gt,
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
const _e = /* @__PURE__ */ Symbol("obsolete");
// @__NO_SIDE_EFFECTS__
function Cn(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && fn();
  var s = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), i = Me(
    /** @type {V} */
    b
  ), u = !v, f = /* @__PURE__ */ new Set();
  return Yn(() => {
    var l = (
      /** @type {Effect} */
      p
    ), o = dt();
    s = o.promise;
    try {
      Promise.resolve(e()).then(o.resolve, (h) => {
        h !== Ee && o.reject(h);
      }).finally(Ne);
    } catch (h) {
      o.reject(h), Ne();
    }
    var _ = (
      /** @type {Batch} */
      g
    );
    if (u) {
      if ((l.f & ce) !== 0)
        var a = bt();
      if (
        // boundary can be null if the async derived is inside an $effect.root not connected to the component render tree
        r.b?.is_rendered()
      )
        _.async_deriveds.get(l)?.reject(_e);
      else
        for (const h of f.values())
          h.reject(_e);
      f.add(o), _.async_deriveds.set(l, o);
    }
    const c = (h, d = void 0) => {
      a?.(), f.delete(o), d !== _e && (_.activate(), d ? (i.f |= K, je(i, d)) : ((i.f & K) !== 0 && (i.f ^= K), je(i, h)), _.deactivate());
    };
    o.promise.then(c, (h) => c(null, h || "unknown"));
  }), Bn(() => {
    for (const l of f)
      l.reject(_e);
  }), new Promise((l) => {
    function o(_) {
      function a() {
        _ === s ? l(i) : o(s);
      }
      _.then(a, a);
    }
    o(s);
  });
}
// @__NO_SIDE_EFFECTS__
function Pn(e) {
  const t = /* @__PURE__ */ Et(e);
  return t.equals = wn, t;
}
function Dn(e) {
  var t = e.effects;
  if (t !== null) {
    e.effects = null;
    for (var n = 0; n < t.length; n += 1)
      L(
        /** @type {Effect} */
        t[n]
      );
  }
}
function Qe(e) {
  var t, n = p, r = e.parent;
  if (!J && r !== null && e.v !== b && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (j | O)) !== 0)
    return pn(), e.v;
  B(r);
  try {
    e.f &= ~re, Dn(e), t = Vt(e);
  } finally {
    B(n);
  }
  return t;
}
function St(e) {
  var t = Qe(e);
  if (!e.equals(t) && (e.wv = Ut(), (!g?.is_fork || e.deps === null) && (g !== null ? (g.capture(e, t, !0), He?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, E);
    return;
  }
  J || (D !== null ? (tt() || g?.is_fork) && D.set(e, t) : Je(e));
}
function Nn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac !== null && Ie(() => {
        t.ac.abort(Ee), t.ac = null;
      }), t.fn !== null && (t.teardown = en), me(t, 0), nt(t));
}
function kt(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && ae(t);
}
let qe = null, ie = null, g = null, He = null, D = null, $e = null, Be = !1, fe = null, Ae = null;
var it = 0;
let Fn = 1;
class W {
  id = Fn++;
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
  #u = /* @__PURE__ */ new Set();
  /**
   * A map of branches that still exist, but will be destroyed when this batch
   * is committed — we skip over these during `process`.
   * The value contains child effects that were dirty/maybe_dirty before being reset,
   * so they can be rescheduled if the branch survives.
   * @type {Map<Effect, { d: Effect[], m: Effect[] }>}
   */
  #a = /* @__PURE__ */ new Map();
  /**
   * Inverse of #skipped_branches which we need to tell prior batches to unskip them when committing
   * @type {Set<Effect>}
   */
  #p = /* @__PURE__ */ new Set();
  is_fork = !1;
  #o = !1;
  constructor() {
    ie === null ? qe = ie = this : (ie.#r = this, this.#v = ie), ie = this;
  }
  #y() {
    if (this.is_fork) return !0;
    for (const r of this.#e.keys()) {
      for (var t = r, n = !1; t.parent !== null; ) {
        if (this.#a.has(t)) {
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
    this.#a.has(t) || this.#a.set(t, { d: [], m: [] }), this.#p.delete(t);
  }
  /**
   * Remove an effect from the #skipped_branches map and reschedule
   * any tracked dirty/maybe_dirty child effects
   * @param {Effect} effect
   * @param {(e: Effect) => void} callback
   */
  unskip_effect(t, n = (r) => this.schedule(r)) {
    var r = this.#a.get(t);
    if (r) {
      this.#a.delete(t);
      for (var s of r.d)
        w(s, m), n(s);
      for (s of r.m)
        w(s, F), n(s);
    }
    this.#p.add(t);
  }
  #g() {
    this.#i = !0, it++ > 1e3 && (this.#_(), jn());
    for (const l of this.#f)
      this.#u.delete(l), w(l, m), this.schedule(l);
    for (const l of this.#u)
      w(l, F), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = fe = [], r = [], s = Ae = [];
    for (const l of t)
      try {
        this.#b(l, n, r);
      } catch (o) {
        throw At(l), this.#y() || this.discard(), o;
      }
    if (g = null, s.length > 0) {
      var i = W.ensure();
      for (const l of s)
        i.schedule(l);
    }
    if (fe = null, Ae = null, this.#y()) {
      this.#h(r), this.#h(n);
      for (const [l, o] of this.#a)
        Tt(l, o);
      s.length > 0 && /** @type {unknown} */
      g.#g();
      return;
    }
    const u = this.#w();
    if (u) {
      this.#h(r), this.#h(n), u.#E(this);
      return;
    }
    this.#f.clear(), this.#u.clear();
    for (const l of this.#c) l(this);
    this.#c.clear(), He = this, lt(r), lt(n), He = null, this.#s?.resolve();
    var f = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      g
    );
    if (this.#l === 0 && (this.#t.length === 0 || f !== null) && this.#_(), this.#t.length > 0)
      if (f !== null) {
        const l = f;
        l.#t.push(...this.#t.filter((o) => !l.#t.includes(o)));
      } else
        f = this;
    f !== null && (M.clear(), f.#g());
  }
  /**
   * Traverse the effect tree, executing effects or stashing
   * them for later execution as appropriate
   * @param {Effect} root
   * @param {Effect[]} effects
   * @param {Effect[]} render_effects
   */
  #b(t, n, r) {
    t.f ^= E;
    for (var s = t.first; s !== null; ) {
      var i = s.f, u = (i & (I | V)) !== 0, f = u && (i & E) !== 0, l = f || (i & O) !== 0 || this.#a.has(s);
      if (!l && s.fn !== null) {
        u ? s.f ^= E : (i & ye) !== 0 ? n.push(s) : ke(s) && ((i & P) !== 0 && this.#u.add(s), ae(s));
        var o = s.first;
        if (o !== null) {
          s = o;
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
  #E(t) {
    for (const [r, s] of t.current)
      !this.previous.has(r) && t.previous.has(r) && this.previous.set(r, t.previous.get(r)), this.current.set(r, s);
    for (const [r, s] of t.async_deriveds) {
      const i = this.async_deriveds.get(r);
      i && s.promise.then(i.resolve).catch(i.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#f, t.#u);
    const n = (r) => {
      var s = r.reactions;
      if (s !== null && !((r.f & S) !== 0 && (r.f & (m | F)) === 0))
        for (const f of s) {
          var i = f.f;
          if ((i & S) !== 0)
            n(
              /** @type {Derived} */
              f
            );
          else {
            var u = (
              /** @type {Effect} */
              f
            );
            i & (ue | P) && !this.async_deriveds.has(u) && (this.#u.delete(u), w(u, m), this.schedule(u));
          }
        }
    };
    for (const r of this.current.keys())
      n(r);
    this.oncommit(() => t.discard()), t.#_(), g = this, this.#g();
  }
  /**
   * @param {Effect[]} effects
   */
  #h(t) {
    for (var n = 0; n < t.length; n += 1)
      mt(t[n], this.#f, this.#u);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== b && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & K) === 0 && (this.current.set(t, [n, r]), D?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    g = this;
  }
  deactivate() {
    g = null, D = null;
  }
  flush() {
    try {
      Be = !0, g = this, this.#g();
    } finally {
      it = 0, $e = null, fe = null, Ae = null, Be = !1, g = null, D = null, M.clear();
    }
  }
  discard() {
    for (const t of this.#n) t(this);
    this.#n.clear();
    for (const t of this.async_deriveds.values())
      t.reject(_e);
    this.#_(), this.#s?.resolve();
  }
  /**
   * @param {Effect} effect
   */
  register_created_effect(t) {
    this.#d.push(t);
  }
  #m() {
    for (let a = qe; a !== null; a = a.#r) {
      var t = a.id < this.id, n = [];
      for (const [c, [h, d]] of this.current) {
        if (a.current.has(c)) {
          var r = (
            /** @type {[any, boolean]} */
            a.current.get(c)[0]
          );
          if (t && h !== r)
            a.current.set(c, [h, d]);
          else
            continue;
        }
        n.push(c);
      }
      if (t)
        for (const [c, h] of this.async_deriveds) {
          const d = a.async_deriveds.get(c);
          d && h.promise.then(d.resolve).catch(d.reject);
        }
      var s = [...a.current.keys()].filter(
        (c) => !/** @type {[any, boolean]} */
        a.current.get(c)[1]
      );
      if (!(!a.#i || s.length === 0)) {
        var i = s.filter((c) => !this.current.has(c));
        if (i.length === 0)
          t && a.discard();
        else if (n.length > 0) {
          if (t)
            for (const c of this.#p)
              a.unskip_effect(c, (h) => {
                (h.f & (P | ue)) !== 0 ? a.schedule(h) : a.#h([h]);
              });
          a.activate();
          var u = /* @__PURE__ */ new Set(), f = /* @__PURE__ */ new Map();
          for (var l of n)
            xt(l, i, u, f);
          f = /* @__PURE__ */ new Map();
          var o = [...a.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (o.length > 0)
            for (const c of this.#d)
              (c.f & (j | O | Pe)) === 0 && Xe(c, o, f) && ((c.f & (ue | P)) !== 0 ? (w(c, m), a.schedule(c)) : a.#f.add(c));
          if (a.#t.length > 0 && !a.#o) {
            a.apply();
            for (var _ of a.#t)
              a.#b(_, [], []);
            a.#t = [];
          }
          a.deactivate();
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
    this.#o || (this.#o = !0, Z(() => {
      this.#o = !1, this.linked && this.flush();
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
      this.#u.add(r);
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
    return (this.#s ??= dt()).promise;
  }
  static ensure() {
    if (g === null) {
      const t = g = new W();
      Be || Z(() => {
        t.#i || t.flush();
      });
    }
    return g;
  }
  apply() {
    {
      D = null;
      return;
    }
  }
  /**
   *
   * @param {Effect} effect
   */
  schedule(t) {
    if ($e = t, t.b?.is_pending && (t.f & (ye | be | vt)) !== 0 && (t.f & ce) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (fe !== null && n === p && (v === null || (v.f & S) === 0))
        return;
      if ((r & (V | I)) !== 0) {
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
      t === null ? qe = n : t.#r = n, n === null ? ie = t : n.#v = t, this.linked = !1;
    }
  }
}
function jn() {
  try {
    cn();
  } catch (e) {
    G(e, $e);
  }
}
let Y = null;
function lt(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (j | O)) === 0 && ke(r) && (Y = /* @__PURE__ */ new Set(), ae(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Mt(r), Y?.size > 0)) {
        M.clear();
        for (const s of Y) {
          if ((s.f & (j | O)) !== 0) continue;
          const i = [s];
          let u = s.parent;
          for (; u !== null; )
            Y.has(u) && (Y.delete(u), i.push(u)), u = u.parent;
          for (let f = i.length - 1; f >= 0; f--) {
            const l = i[f];
            (l.f & (j | O)) === 0 && ae(l);
          }
        }
        Y.clear();
      }
    }
    Y = null;
  }
}
function xt(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const s of e.reactions) {
      const i = s.f;
      (i & S) !== 0 ? xt(
        /** @type {Derived} */
        s,
        t,
        n,
        r
      ) : (i & (ue | P)) !== 0 && (i & m) === 0 && Xe(s, t, r) && (w(s, m), Ze(
        /** @type {Effect} */
        s
      ));
    }
}
function Xe(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const s of e.deps) {
      if (Ce.call(t, s))
        return !0;
      if ((s.f & S) !== 0 && Xe(
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
function Ze(e) {
  g.schedule(e);
}
function Tt(e, t) {
  if (!((e.f & I) !== 0 && (e.f & E) !== 0)) {
    (e.f & m) !== 0 ? t.d.push(e) : (e.f & F) !== 0 && t.m.push(e), w(e, E);
    for (var n = e.first; n !== null; )
      Tt(n, t), n = n.next;
  }
}
function At(e) {
  w(e, E);
  for (var t = e.first; t !== null; )
    At(t), t = t.next;
}
let Fe = /* @__PURE__ */ new Set();
const M = /* @__PURE__ */ new Map();
let Rt = !1;
function Me(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: gt,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function $(e, t) {
  const n = Me(e);
  return Kn(n), n;
}
function X(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!N || (v.f & Pe) !== 0) && Se() && (v.f & (S | P | ue | Pe)) !== 0 && (q === null || !q.has(e)) && dn();
  let r = n ? de(t) : t;
  return je(e, r, Ae);
}
function je(e, t, n = null) {
  if (!e.equals(t)) {
    J ? M.set(e, t) : M.has(e) || M.set(e, e.v);
    var r = W.ensure();
    if (r.capture(e, t), (e.f & S) !== 0) {
      const s = (
        /** @type {Derived} */
        e
      );
      (e.f & m) !== 0 && Qe(s), D === null && Je(s);
    }
    e.wv = Ut(), Ot(e, m, n), Se() && p !== null && (p.f & E) !== 0 && (p.f & (I | V)) === 0 && (A === null ? Wn([e]) : A.push(e)), !r.is_fork && Fe.size > 0 && !Rt && In();
  }
  return t;
}
function In() {
  Rt = !1;
  for (const e of Fe) {
    (e.f & E) !== 0 && w(e, F);
    let t;
    try {
      t = ke(e);
    } catch {
      t = !0;
    }
    t && ae(e);
  }
  Fe.clear();
}
function ge(e) {
  X(e, e.v + 1);
}
function Ot(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var s = Se(), i = r.length, u = 0; u < i; u++) {
      var f = r[u], l = f.f;
      if (!(!s && f === p)) {
        var o = (l & m) === 0;
        if (o && w(f, t), (l & Pe) !== 0)
          Fe.add(
            /** @type {Effect} */
            f
          );
        else if ((l & S) !== 0) {
          var _ = (
            /** @type {Derived} */
            f
          );
          D?.delete(_), (l & re) === 0 && (l & R && (p === null || (p.f & De) === 0) && (f.f |= re), Ot(_, F, n));
        } else if (o) {
          var a = (
            /** @type {Effect} */
            f
          );
          (l & P) !== 0 && Y !== null && Y.add(a), n !== null ? n.push(a) : Ze(a);
        }
      }
    }
}
function de(e) {
  if (typeof e != "object" || e === null || pe in e)
    return e;
  const t = _t(e);
  if (t !== Xt && t !== Zt)
    return e;
  var n = /* @__PURE__ */ new Map(), r = Gt(e), s = /* @__PURE__ */ $(0), i = ne, u = (f) => {
    if (ne === i)
      return f();
    var l = v, o = ne;
    C(null), ot(i);
    var _ = f();
    return C(l), ot(o), _;
  };
  return r && n.set("length", /* @__PURE__ */ $(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(f, l, o) {
        (!("value" in o) || o.configurable === !1 || o.enumerable === !1 || o.writable === !1) && hn();
        var _ = n.get(l);
        return _ === void 0 ? u(() => {
          var a = /* @__PURE__ */ $(o.value);
          return n.set(l, a), a;
        }) : X(_, o.value, !0), !0;
      },
      deleteProperty(f, l) {
        var o = n.get(l);
        if (o === void 0) {
          if (l in f) {
            const _ = u(() => /* @__PURE__ */ $(b));
            n.set(l, _), ge(s);
          }
        } else
          X(o, b), ge(s);
        return !0;
      },
      get(f, l, o) {
        if (l === pe)
          return e;
        var _ = n.get(l), a = l in f;
        if (_ === void 0 && (!a || ve(f, l)?.writable) && (_ = u(() => {
          var h = de(a ? f[l] : b), d = /* @__PURE__ */ $(h);
          return d;
        }), n.set(l, _)), _ !== void 0) {
          var c = te(_);
          return c === b ? void 0 : c;
        }
        return Reflect.get(f, l, o);
      },
      getOwnPropertyDescriptor(f, l) {
        var o = Reflect.getOwnPropertyDescriptor(f, l);
        if (o && "value" in o) {
          var _ = n.get(l);
          _ && (o.value = te(_));
        } else if (o === void 0) {
          var a = n.get(l), c = a?.v;
          if (a !== void 0 && c !== b)
            return {
              enumerable: !0,
              configurable: !0,
              value: c,
              writable: !0
            };
        }
        return o;
      },
      has(f, l) {
        if (l === pe)
          return !0;
        var o = n.get(l), _ = o !== void 0 && o.v !== b || Reflect.has(f, l);
        if (o !== void 0 || p !== null && (!_ || ve(f, l)?.writable)) {
          o === void 0 && (o = u(() => {
            var c = _ ? de(f[l]) : b, h = /* @__PURE__ */ $(c);
            return h;
          }), n.set(l, o));
          var a = te(o);
          if (a === b)
            return !1;
        }
        return _;
      },
      set(f, l, o, _) {
        var a = n.get(l), c = l in f;
        if (r && l === "length")
          for (var h = o; h < /** @type {Source<number>} */
          a.v; h += 1) {
            var d = n.get(h + "");
            d !== void 0 ? X(d, b) : h in f && (d = u(() => /* @__PURE__ */ $(b)), n.set(h + "", d));
          }
        if (a === void 0)
          (!c || ve(f, l)?.writable) && (a = u(() => /* @__PURE__ */ $(void 0)), X(a, de(o)), n.set(l, a));
        else {
          c = a.v !== b;
          var y = u(() => de(o));
          X(a, y);
        }
        var H = Reflect.getOwnPropertyDescriptor(f, l);
        if (H?.set && H.set.call(_, o), !c) {
          if (r && typeof l == "string") {
            var U = (
              /** @type {Source<number>} */
              n.get("length")
            ), se = Number(l);
            Number.isInteger(se) && se >= U.v && X(U, se + 1);
          }
          ge(s);
        }
        return !0;
      },
      ownKeys(f) {
        te(s);
        var l = Reflect.ownKeys(f).filter((a) => {
          var c = n.get(a);
          return c === void 0 || c.v !== b;
        });
        for (var [o, _] of n)
          _.v !== b && !(o in f) && l.push(o);
        return l;
      },
      setPrototypeOf() {
        _n();
      }
    }
  );
}
var ft, Mn, Ct, Pt;
function Ln() {
  if (ft === void 0) {
    ft = window, Mn = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    Ct = ve(t, "firstChild").get, Pt = ve(t, "nextSibling").get, rt(e) && (e[sn] = void 0, e[rn] = null, e[ln] = void 0, e.__e = void 0), rt(n) && (n[ze] = void 0);
  }
}
function Dt(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function Nt(e) {
  return (
    /** @type {TemplateNode | null} */
    Ct.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function et(e) {
  return (
    /** @type {TemplateNode | null} */
    Pt.call(e)
  );
}
function cr(e, t) {
  return /* @__PURE__ */ Nt(e);
}
function hr(e, t = !1) {
  {
    var n = /* @__PURE__ */ Nt(e);
    return n instanceof Comment && n.data === "" ? /* @__PURE__ */ et(n) : n;
  }
}
function _r() {
  return !1;
}
function dr(e, t, n) {
  return (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  );
}
function Ft(e) {
  p === null && (v === null && an(), on()), J && un();
}
function qn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function z(e, t) {
  var n = p;
  n !== null && (n.f & O) !== 0 && (e |= O);
  var r = {
    ctx: k,
    deps: null,
    nodes: null,
    f: e | m | R,
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
  g?.register_created_effect(r);
  var s = r;
  if ((e & ye) !== 0)
    fe !== null ? fe.push(r) : W.ensure().schedule(r);
  else if (t !== null) {
    try {
      ae(r);
    } catch (u) {
      throw L(r), u;
    }
    s.deps === null && s.teardown === null && s.nodes === null && s.first === s.last && // either `null`, or a singular child
    (s.f & he) === 0 && (s = s.first, (e & P) !== 0 && (e & we) !== 0 && s !== null && (s.f |= we));
  }
  if (s !== null && (s.parent = n, n !== null && qn(s, n), v !== null && (v.f & S) !== 0 && (e & V) === 0)) {
    var i = (
      /** @type {Derived} */
      v
    );
    (i.effects ??= []).push(s);
  }
  return r;
}
function tt() {
  return v !== null && !N;
}
function Bn(e) {
  const t = z(be, null);
  return w(t, E), t.teardown = e, t;
}
function vr(e) {
  Ft();
  var t = (
    /** @type {Effect} */
    p.f
  ), n = !v && (t & I) !== 0 && k !== null && !k.i;
  if (n) {
    var r = (
      /** @type {ComponentContext} */
      k
    );
    (r.e ??= []).push(e);
  } else
    return jt(e);
}
function jt(e) {
  return z(ye | pt, e);
}
function pr(e) {
  return Ft(), z(be | pt, e);
}
function Un(e) {
  W.ensure();
  const t = z(V | he, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? Re(t, () => {
      L(t), r(void 0);
    }) : (L(t), r(void 0));
  });
}
function Yn(e) {
  return z(ue | he, e);
}
function Vn(e, t = 0) {
  return z(be | t, e);
}
function gr(e, t = [], n = [], r = []) {
  Rn(r, t, n, (s) => {
    z(be, () => {
      e(...s.map(te));
    });
  });
}
function zn(e, t = 0) {
  var n = z(P | t, e);
  return n;
}
function Q(e) {
  return z(I | he, e);
}
function It(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = J, r = v;
    ut(!0), C(null);
    try {
      t.call(null);
    } finally {
      ut(n), C(r);
    }
  }
}
function nt(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const s = n.ac;
    s !== null && Ie(() => {
      s.abort(Ee);
    });
    var r = n.next;
    (n.f & V) !== 0 ? n.parent = null : L(n, t), n = r;
  }
}
function Hn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & I) === 0 && L(t), t = n;
  }
}
function L(e, t = !0) {
  var n = !1;
  (t || (e.f & nn) !== 0) && e.nodes !== null && e.nodes.end !== null && ($n(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= st, nt(e, t && !n), me(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const i of r)
      i.stop();
  It(e), e.f ^= st, e.f |= j;
  var s = e.parent;
  s !== null && s.first !== null && Mt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function $n(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ et(e);
    e.remove(), e = n;
  }
}
function Mt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function Re(e, t, n = !0) {
  var r = [];
  Lt(e, r, !0);
  var s = () => {
    n && L(e), t && t();
  }, i = r.length;
  if (i > 0) {
    var u = () => --i || s();
    for (var f of r)
      f.out(u);
  } else
    s();
}
function Lt(e, t, n) {
  if ((e.f & O) === 0) {
    e.f ^= O;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const f of r)
        (f.is_global || n) && t.push(f);
    for (var s = e.first; s !== null; ) {
      var i = s.next;
      if ((s.f & V) === 0) {
        var u = (s.f & we) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (s.f & I) !== 0 && (e.f & P) !== 0;
        Lt(s, t, u ? n : !1);
      }
      s = i;
    }
  }
}
function yr(e) {
  qt(e, !0);
}
function qt(e, t) {
  if ((e.f & O) !== 0) {
    e.f ^= O, (e.f & E) === 0 && (w(e, m), W.ensure().schedule(e));
    for (var n = e.first; n !== null; ) {
      var r = n.next, s = (n.f & we) !== 0 || (n.f & I) !== 0;
      qt(n, s ? t : !1), n = r;
    }
    var i = e.nodes && e.nodes.t;
    if (i !== null)
      for (const u of i)
        (u.is_global || t) && u.in();
  }
}
function Gn(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var s = n === r ? null : /* @__PURE__ */ et(n);
      t.append(n), n = s;
    }
}
let Oe = !1, J = !1;
function ut(e) {
  J = e;
}
let v = null, N = !1;
function C(e) {
  v = e;
}
let p = null;
function B(e) {
  p = e;
}
let q = null;
function Kn(e) {
  v !== null && (q ??= /* @__PURE__ */ new Set()).add(e);
}
let x = null, T = 0, A = null;
function Wn(e) {
  A = e;
}
let Bt = 1, ee = 0, ne = ee;
function ot(e) {
  ne = e;
}
function Ut() {
  return ++Bt;
}
function ke(e) {
  var t = e.f;
  if ((t & m) !== 0)
    return !0;
  if (t & S && (e.f &= ~re), (t & F) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, s = 0; s < r; s++) {
      var i = n[s];
      if (ke(
        /** @type {Derived} */
        i
      ) && St(
        /** @type {Derived} */
        i
      ), i.wv > e.wv)
        return !0;
    }
    (t & R) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    D === null && w(e, E);
  }
  return !1;
}
function Yt(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(q !== null && q.has(e)))
    for (var s = 0; s < r.length; s++) {
      var i = r[s];
      (i.f & S) !== 0 ? Yt(
        /** @type {Derived} */
        i,
        t,
        !1
      ) : t === i && (n ? w(i, m) : (i.f & E) !== 0 && w(i, F), Ze(
        /** @type {Effect} */
        i
      ));
    }
}
function Vt(e) {
  var t = x, n = T, r = A, s = v, i = q, u = k, f = N, l = ne, o = e.f;
  x = /** @type {null | Value[]} */
  null, T = 0, A = null, v = (o & (I | V)) === 0 ? e : null, q = null, oe(e.ctx), N = !1, ne = ++ee, e.ac !== null && (Ie(() => {
    e.ac.abort(Ee);
  }), e.ac = null);
  try {
    e.f |= De;
    var _ = (
      /** @type {Function} */
      e.fn
    ), a = _();
    e.f |= ce;
    var c = e.deps, h = g?.is_fork;
    if (x !== null) {
      var d;
      if (h || me(e, T), c !== null && T > 0)
        for (c.length = T + x.length, d = 0; d < x.length; d++)
          c[T + d] = x[d];
      else
        e.deps = c = x;
      if (tt() && (e.f & R) !== 0)
        for (d = T; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && T < c.length && (me(e, T), c.length = T);
    if (Se() && A !== null && !N && c !== null && (e.f & (S | F | m)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      A.length; d++)
        Yt(
          A[d],
          /** @type {Effect} */
          e
        );
    if (s !== null && s !== e) {
      if (ee++, s.deps !== null)
        for (let y = 0; y < n; y += 1)
          s.deps[y].rv = ee;
      if (t !== null)
        for (const y of t)
          y.rv = ee;
      A !== null && (r === null ? r = A : r.push(.../** @type {Source[]} */
      A));
    }
    return (e.f & K) !== 0 && (e.f ^= K), a;
  } catch (y) {
    return yt(y);
  } finally {
    e.f ^= De, x = t, T = n, A = r, v = s, q = i, oe(u), N = f, ne = l;
  }
}
function Jn(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = Kt.call(n, e);
    if (r !== -1) {
      var s = n.length - 1;
      s === 0 ? n = t.reactions = null : (n[r] = n[s], n.pop());
    }
  }
  if (n === null && (t.f & S) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (x === null || !Ce.call(x, t))) {
    var i = (
      /** @type {Derived} */
      t
    );
    (i.f & R) !== 0 && (i.f ^= R, i.f &= ~re), i.v !== b && Je(i), i.ac !== null && Ie(() => {
      i.ac.abort(Ee), i.ac = null, w(i, m);
    }), Nn(i), me(i, 0);
  }
}
function me(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      Jn(e, n[r]);
}
function ae(e) {
  var t = e.f;
  if ((t & j) === 0) {
    w(e, E);
    var n = p, r = Oe;
    p = e, Oe = (t & (I | V)) === 0;
    try {
      (t & (P | vt)) !== 0 ? Hn(e) : nt(e), It(e);
      var s = Vt(e);
      e.teardown = typeof s == "function" ? s : null, e.wv = Bt;
      var i;
      ht && $t && (e.f & m) !== 0 && e.deps;
    } finally {
      Oe = r, p = n;
    }
  }
}
function te(e) {
  var t = e.f, n = (t & S) !== 0;
  if (v !== null && !N) {
    var r = p !== null && (p.f & j) !== 0;
    if (!r && (q === null || !q.has(e))) {
      var s = v.deps;
      if ((v.f & De) !== 0)
        e.rv < ee && (e.rv = ee, x === null && s !== null && s[T] === e ? T++ : x === null ? x = [e] : x.push(e));
      else {
        v.deps ??= [], Ce.call(v.deps, e) || v.deps.push(e);
        var i = e.reactions;
        i === null ? e.reactions = [v] : Ce.call(i, v) || i.push(v);
      }
    }
  }
  if (J && M.has(e))
    return M.get(e);
  if (n) {
    var u = (
      /** @type {Derived} */
      e
    );
    if (J) {
      var f = u.v;
      return ((u.f & E) === 0 && u.reactions !== null || Ht(u)) && (f = Qe(u)), M.set(u, f), f;
    }
    var l = (u.f & R) === 0 && !N && v !== null && (Oe || (v.f & R) !== 0), o = (u.f & ce) === 0;
    ke(u) && (l && (u.f |= R), St(u)), l && !o && (kt(u), zt(u));
  }
  if (D?.has(e))
    return D.get(e);
  if ((e.f & K) !== 0)
    throw e.v;
  return e.v;
}
function zt(e) {
  if (e.f |= R, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & S) !== 0 && (t.f & R) === 0 && (kt(
        /** @type {Derived} */
        t
      ), zt(
        /** @type {Derived} */
        t
      ));
}
function Ht(e) {
  if (e.v === b) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (M.has(t) || (t.f & S) !== 0 && Ht(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function Qn(e) {
  var t = N;
  try {
    return N = !0, e();
  } finally {
    N = t;
  }
}
function wr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (pe in e)
      Ge(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && pe in n && Ge(n);
      }
  }
}
function Ge(e, t = /* @__PURE__ */ new Set()) {
  if (typeof e == "object" && e !== null && // We don't want to traverse DOM elements
  !(e instanceof EventTarget) && !t.has(e)) {
    t.add(e), e instanceof Date && e.getTime();
    for (let r in e)
      try {
        Ge(e[r], t);
      } catch {
      }
    const n = _t(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = Qt(n);
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
const xe = /* @__PURE__ */ Symbol("events"), Xn = /* @__PURE__ */ new Set(), at = /* @__PURE__ */ new Set();
let Ue = null, Ye = !1;
function ct(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, s = e.composedPath?.() || [], i = (
    /** @type {null | Element} */
    s[0] || e.target
  );
  Ue = e, Ye || (Ye = !0, setTimeout(() => {
    Ye = !1, Ue = null;
  }));
  var u = 0, f = Ue === e && e[xe];
  if (f) {
    var l = s.indexOf(f);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[xe] = t;
      return;
    }
    var o = s.indexOf(t);
    if (o === -1)
      return;
    l <= o && (u = l);
  }
  if (i = /** @type {Element} */
  s[u] || e.target, i !== t) {
    Jt(e, "currentTarget", {
      configurable: !0,
      get() {
        return i || n;
      }
    });
    var _ = v, a = p;
    C(null), B(null);
    try {
      for (var c, h = []; i !== null && i !== t; ) {
        try {
          var d = i[xe]?.[r];
          d != null && (!/** @type {any} */
          i.disabled || // DOM could've been updated already by the time this is reached, so we check this as well
          // -> the target could not have been disabled because it emits the event in the first place
          e.target === i) && d.call(i, e);
        } catch (y) {
          c ? h.push(y) : c = y;
        }
        if (e.cancelBubble) break;
        u++, i = u < s.length ? (
          /** @type {Element} */
          s[u]
        ) : null;
      }
      if (c) {
        for (let y of h)
          queueMicrotask(() => {
            throw y;
          });
        throw c;
      }
    } finally {
      e[xe] = t, delete e.currentTarget, C(_), B(a);
    }
  }
}
const Zn = ["touchstart", "touchmove"];
function er(e) {
  return Zn.includes(e);
}
function mr(e, t) {
  var n = t == null ? "" : typeof t == "object" ? `${t}` : t;
  n !== /** @type {any} */
  (e[ze] ??= e.nodeValue) && (e[ze] = n, e.nodeValue = `${n}`);
}
function br(e, t) {
  return tr(e, t);
}
const Te = /* @__PURE__ */ new Map();
function tr(e, { target: t, anchor: n, props: r = {}, events: s, context: i, intro: u = !0, transformError: f }) {
  Ln();
  var l = void 0, o = Un(() => {
    var _ = n ?? t.appendChild(Dt());
    Tn(
      /** @type {TemplateNode} */
      _,
      {
        pending: () => {
        }
      },
      (h) => {
        mn({});
        var d = (
          /** @type {ComponentContext} */
          k
        );
        i && (d.c = i), s && (r.$$events = s), l = e(h, r) || {}, bn();
      },
      f
    );
    var a = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var y = h[d];
        if (!a.has(y)) {
          a.add(y);
          var H = er(y);
          for (const Le of [t, document]) {
            var U = Te.get(Le);
            U === void 0 && (U = /* @__PURE__ */ new Map(), Te.set(Le, U));
            var se = U.get(y);
            se === void 0 ? (Le.addEventListener(y, ct, { passive: H }), U.set(y, 1)) : U.set(y, se + 1);
          }
        }
      }
    };
    return c(Wt(Xn)), at.add(c), () => {
      for (var h of a)
        for (const H of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            Te.get(H)
          ), y = (
            /** @type {number} */
            d.get(h)
          );
          --y == 0 ? (H.removeEventListener(h, ct), d.delete(h), d.size === 0 && Te.delete(H)) : d.set(h, y);
        }
      at.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return Ke.set(l, o), l;
}
let Ke = /* @__PURE__ */ new WeakMap();
function Er(e, t) {
  const n = Ke.get(e);
  return n ? (Ke.delete(e), n(t)) : Promise.resolve();
}
export {
  X as A,
  We as B,
  sn as C,
  j as D,
  we as E,
  rr as F,
  ir as G,
  Pn as H,
  J as I,
  hr as J,
  bn as K,
  ur as L,
  mn as M,
  gr as N,
  mr as O,
  sr as P,
  cr as Q,
  br as R,
  pe as S,
  lr as T,
  Er as U,
  Dt as a,
  p as b,
  dr as c,
  L as d,
  nr as e,
  Q as f,
  Nt as g,
  g as h,
  Mn as i,
  zn as j,
  k,
  vr as l,
  Gn as m,
  tn as n,
  Qn as o,
  Re as p,
  fr as q,
  yr as r,
  _r as s,
  te as t,
  pr as u,
  wr as v,
  Et as w,
  ve as x,
  ar as y,
  de as z
};
