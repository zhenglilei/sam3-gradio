let He = !1, Bt = !1;
function Xn() {
  He = !0;
}
const Zn = 2, Jn = 4, Qn = 8, er = 2, m = /* @__PURE__ */ Symbol("uninitialized"), ot = !1;
var Ut = Array.isArray, Yt = Array.prototype.indexOf, Ae = Array.prototype.includes, Vt = Array.from, $t = Object.defineProperty, ve = Object.getOwnPropertyDescriptor, zt = Object.getOwnPropertyDescriptors, Ht = Object.prototype, Gt = Array.prototype, at = Object.getPrototypeOf, Qe = Object.isExtensible;
const Kt = () => {
};
function Wt(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function ct() {
  var e, t, n = new Promise((r, i) => {
    e = r, t = i;
  });
  return { promise: n, resolve: e, reject: t };
}
const S = 2, ye = 4, Fe = 8, ht = 1 << 24, P = 16, z = 32, H = 64, Be = 128, R = 512, b = 1024, E = 2048, M = 4096, F = 8192, N = 16384, ce = 32768, et = 1 << 25, Re = 65536, Oe = 1 << 17, Xt = 1 << 18, he = 1 << 19, Zt = 1 << 20, ee = 65536, Pe = 1 << 21, fe = 1 << 22, $ = 1 << 23, pe = /* @__PURE__ */ Symbol("$state"), tr = /* @__PURE__ */ Symbol("legacy props"), Jt = /* @__PURE__ */ Symbol("attributes"), Qt = /* @__PURE__ */ Symbol("class"), en = /* @__PURE__ */ Symbol("style"), Ue = /* @__PURE__ */ Symbol("text"), je = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}();
function tn() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function nn() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function rn() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function sn() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function ln() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function fn() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function un() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function on() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function _t(e) {
  return e === this.v;
}
function an(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function cn(e) {
  return !an(e, this.v);
}
let T = null;
function oe(e) {
  T = e;
}
function hn(e, t = !1, n) {
  T = {
    p: T,
    i: !1,
    c: null,
    e: null,
    s: e,
    x: null,
    r: (
      /** @type {Effect} */
      p
    ),
    l: He && !t ? { s: null, u: null, $: [] } : null
  };
}
function _n(e) {
  var t = (
    /** @type {ComponentContext} */
    T
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      Fn(r);
  }
  return t.i = !0, T = t.p, /** @type {T} */
  {};
}
function me() {
  return !He || T !== null && T.l === null;
}
let se = [];
function dn() {
  var e = se;
  se = [], Wt(e);
}
function ue(e) {
  if (se.length === 0) {
    var t = se;
    queueMicrotask(() => {
      t === se && dn();
    });
  }
  se.push(e);
}
function dt(e) {
  var t = p;
  if (t === null)
    return v.f |= $, e;
  if ((t.f & ce) === 0 && (t.f & ye) === 0)
    throw e;
  V(e, t);
}
function V(e, t) {
  if (!(t !== null && (t.f & N) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & Be) !== 0) {
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
const vn = -7169;
function w(e, t) {
  e.f = e.f & vn | t;
}
function Ge(e) {
  (e.f & R) !== 0 || e.deps === null ? w(e, b) : w(e, M);
}
function vt(e) {
  if (e !== null)
    for (const t of e)
      (t.f & S) === 0 || (t.f & ee) === 0 || (t.f ^= ee, vt(
        /** @type {Derived} */
        t.deps
      ));
}
function pt(e, t, n) {
  (e.f & E) !== 0 ? t.add(e) : (e.f & M) !== 0 && n.add(e), vt(e.deps), w(e, b);
}
function pn(e) {
  let t = 0, n = Ie(0), r;
  return () => {
    Ze() && (Z(n), Mn(() => (t === 0 && (r = zn(() => e(() => ge(n)))), t += 1, () => {
      ue(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, ge(n));
      });
    })));
  };
}
var gn = Re | he;
function yn(e, t, n, r) {
  new wn(e, t, n, r);
}
class wn {
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
  #s;
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
  #i = null;
  /** @type {DocumentFragment | null} */
  #t = null;
  #_ = 0;
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
  #y = pn(() => (this.#o = Ie(this.#_), () => {
    this.#o = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(t, n, r, i) {
    this.#s = t, this.#r = n, this.#h = (s) => {
      var o = (
        /** @type {Effect} */
        p
      );
      o.b = this, o.f |= Be, r(s);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = i ?? this.parent?.transform_error ?? ((s) => s), this.#n = Ln(() => {
      this.#w();
    }, gn);
  }
  #g() {
    try {
      this.#l = K(() => this.#h(this.#s));
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #b(t) {
    const n = this.#r.failed;
    n && (this.#i = K(() => {
      n(
        this.#s,
        () => t,
        () => () => {
        }
      );
    }));
  }
  #E() {
    const t = this.#r.pending;
    t && (this.is_pending = !0, this.#e = K(() => t(this.#s)), ue(() => {
      var n = this.#t = document.createDocumentFragment(), r = Rt();
      n.append(r), this.#l = this.#m(() => K(() => this.#h(r))), this.#f === 0 && (this.#s.before(n), this.#t = null, xe(
        /** @type {Effect} */
        this.#e,
        () => {
          this.#e = null;
        }
      ), this.#c(
        /** @type {Batch} */
        g
      ));
    }));
  }
  #w() {
    try {
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#_ = 0, this.#l = K(() => {
        this.#h(this.#s);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        Un(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = K(() => n(this.#s));
      } else
        this.#c(
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
  #c(t) {
    this.is_pending = !1, t.transfer_effects(this.#a, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(t) {
    pt(t, this.#a, this.#p);
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
  #m(t) {
    var n = p, r = v, i = T;
    L(this.#n), O(this.#n), oe(this.#n.ctx);
    try {
      return te.ensure(), t();
    } catch (s) {
      return dt(s), null;
    } finally {
      L(n), O(r), oe(i);
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
    this.#f += t, this.#f === 0 && (this.#c(n), this.#e && xe(this.#e, () => {
      this.#e = null;
    }), this.#t && (this.#s.before(this.#t), this.#t = null));
  }
  /**
   * Update the source that powers `$effect.pending()` inside this boundary,
   * and controls when the current `pending` snippet (if any) is removed.
   * Do not call from inside the class
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  update_pending_count(t, n) {
    this.#d(t, n), this.#_ += t, !(!this.#o || this.#u) && (this.#u = !0, ue(() => {
      this.#u = !1, this.#o && Ne(this.#o, this.#_);
    }));
  }
  get_effect_pending() {
    return this.#y(), Z(
      /** @type {Source<number>} */
      this.#o
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    g?.is_fork ? (this.#l && g.skip_effect(this.#l), this.#e && g.skip_effect(this.#e), this.#i && g.skip_effect(this.#i), g.oncommit(() => {
      this.#S(t);
    })) : this.#S(t);
  }
  /**
   * @param {unknown} error
   */
  #S(t) {
    this.#l && (j(this.#l), this.#l = null), this.#e && (j(this.#e), this.#e = null), this.#i && (j(this.#i), this.#i = null);
    var n = this.#r.onerror;
    let r = this.#r.failed;
    var i = !1, s = !1;
    const o = () => {
      if (i) {
        on();
        return;
      }
      i = !0, s && fn(), this.#i !== null && xe(this.#i, () => {
        this.#i = null;
      }), this.#m(() => {
        this.#w();
      });
    }, u = (l) => {
      try {
        s = !0, n?.(l, o), s = !1;
      } catch (f) {
        V(f, this.#n && this.#n.parent);
      }
      r && (this.#i = this.#m(() => {
        try {
          return K(() => {
            var f = (
              /** @type {Effect} */
              p
            );
            f.b = this, f.f |= Be, r(
              this.#s,
              () => l,
              () => o
            );
          });
        } catch (f) {
          return V(
            f,
            /** @type {Effect} */
            this.#n.parent
          ), null;
        }
      }));
    };
    ue(() => {
      var l;
      try {
        l = this.transform_error(t);
      } catch (f) {
        V(f, this.#n && this.#n.parent);
        return;
      }
      l !== null && typeof l == "object" && typeof /** @type {any} */
      l.then == "function" ? l.then(
        u,
        /** @param {unknown} e */
        (f) => V(f, this.#n && this.#n.parent)
      ) : u(l);
    });
  }
}
function mn(e, t, n, r) {
  const i = me() ? yt : Sn;
  var s = e.filter((h) => !h.settled), o = t.map(i);
  if (n.length === 0 && s.length === 0) {
    r(o);
    return;
  }
  var u = (
    /** @type {Effect} */
    p
  ), l = bn(), f = s.length === 1 ? s[0].promise : s.length > 1 ? Promise.all(s.map((h) => h.promise)) : null;
  function _(h) {
    if ((u.f & N) === 0) {
      l();
      try {
        r([...o, ...h]);
      } catch (d) {
        V(d, u);
      }
      Ce();
    }
  }
  var a = gt();
  if (n.length === 0) {
    f.then(() => _([])).finally(a);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ En(h))).then(_).catch((h) => V(h, u)).finally(a);
  }
  f ? f.then(() => {
    l(), c(), Ce();
  }) : c();
}
function bn() {
  var e = (
    /** @type {Effect} */
    p
  ), t = v, n = T, r = (
    /** @type {Batch} */
    g
  );
  return function(s = !0) {
    L(e), O(t), oe(n), s && (e.f & N) === 0 && (r?.activate(), r?.apply());
  };
}
function Ce(e = !0) {
  L(null), O(null), oe(null), e && g?.deactivate();
}
function gt() {
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
function yt(e) {
  var t = S | E;
  return p !== null && (p.f |= he), {
    ctx: T,
    deps: null,
    effects: null,
    equals: _t,
    f: t,
    fn: e,
    reactions: null,
    rv: 0,
    v: (
      /** @type {V} */
      m
    ),
    wv: 0,
    parent: p,
    ac: null
  };
}
const _e = /* @__PURE__ */ Symbol("obsolete");
// @__NO_SIDE_EFFECTS__
function En(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && tn();
  var i = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), s = Ie(
    /** @type {V} */
    m
  ), o = !v, u = /* @__PURE__ */ new Set();
  return In(() => {
    var l = (
      /** @type {Effect} */
      p
    ), f = ct();
    i = f.promise;
    try {
      Promise.resolve(e()).then(f.resolve, (h) => {
        h !== je && f.reject(h);
      }).finally(Ce);
    } catch (h) {
      f.reject(h), Ce();
    }
    var _ = (
      /** @type {Batch} */
      g
    );
    if (o) {
      if ((l.f & ce) !== 0)
        var a = gt();
      if (
        // boundary can be null if the async derived is inside an $effect.root not connected to the component render tree
        r.b?.is_rendered()
      )
        _.async_deriveds.get(l)?.reject(_e);
      else
        for (const h of u.values())
          h.reject(_e);
      u.add(f), _.async_deriveds.set(l, f);
    }
    const c = (h, d = void 0) => {
      a?.(), u.delete(f), d !== _e && (_.activate(), d ? (s.f |= $, Ne(s, d)) : ((s.f & $) !== 0 && (s.f ^= $), Ne(s, h)), _.deactivate());
    };
    f.promise.then(c, (h) => c(null, h || "unknown"));
  }), Nn(() => {
    for (const l of u)
      l.reject(_e);
  }), new Promise((l) => {
    function f(_) {
      function a() {
        _ === i ? l(s) : f(i);
      }
      _.then(a, a);
    }
    f(i);
  });
}
// @__NO_SIDE_EFFECTS__
function Sn(e) {
  const t = /* @__PURE__ */ yt(e);
  return t.equals = cn, t;
}
function kn(e) {
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
function Ke(e) {
  var t, n = p, r = e.parent;
  if (!ne && r !== null && e.v !== m && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (N | F)) !== 0)
    return un(), e.v;
  L(r);
  try {
    e.f &= ~ee, kn(e), t = Mt(e);
  } finally {
    L(n);
  }
  return t;
}
function wt(e) {
  var t = Ke(e);
  if (!e.equals(t) && (e.wv = jt(), (!g?.is_fork || e.deps === null) && (g !== null ? (g.capture(e, t, !0), Ye?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, b);
    return;
  }
  ne || (C !== null ? (Ze() || g?.is_fork) && C.set(e, t) : Ge(e));
}
function xn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac?.abort(je), t.fn !== null && (t.teardown = Kt), t.ac = null, we(t, 0), Je(t));
}
function mt(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && ae(t);
}
let Le = null, ie = null, g = null, Ye = null, C = null, Ve = null, qe = !1, le = null, ke = null;
var tt = 0;
let Tn = 1;
class te {
  id = Tn++;
  /** True as soon as `#process` was called */
  #s = !1;
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
  #i = null;
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
    ie === null ? Le = ie = this : (ie.#r = this, this.#v = ie), ie = this;
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
      for (var i of r.d)
        w(i, E), n(i);
      for (i of r.m)
        w(i, M), n(i);
    }
    this.#p.add(t);
  }
  #g() {
    this.#s = !0, tt++ > 1e3 && (this.#d(), An());
    for (const l of this.#f)
      this.#u.delete(l), w(l, E), this.schedule(l);
    for (const l of this.#u)
      w(l, M), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = le = [], r = [], i = ke = [];
    for (const l of t)
      try {
        this.#b(l, n, r);
      } catch (f) {
        throw St(l), this.#y() || this.discard(), f;
      }
    if (g = null, i.length > 0) {
      var s = te.ensure();
      for (const l of i)
        s.schedule(l);
    }
    if (le = null, ke = null, this.#y()) {
      this.#c(r), this.#c(n);
      for (const [l, f] of this.#a)
        Et(l, f);
      i.length > 0 && /** @type {unknown} */
      g.#g();
      return;
    }
    const o = this.#E();
    if (o) {
      this.#c(r), this.#c(n), o.#w(this);
      return;
    }
    this.#f.clear(), this.#u.clear();
    for (const l of this.#h) l(this);
    this.#h.clear(), Ye = this, nt(r), nt(n), Ye = null, this.#i?.resolve();
    var u = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      g
    );
    if (this.#l === 0 && (this.#t.length === 0 || u !== null) && this.#d(), this.#t.length > 0)
      if (u !== null) {
        const l = u;
        l.#t.push(...this.#t.filter((f) => !l.#t.includes(f)));
      } else
        u = this;
    u !== null && u.#g();
  }
  /**
   * Traverse the effect tree, executing effects or stashing
   * them for later execution as appropriate
   * @param {Effect} root
   * @param {Effect[]} effects
   * @param {Effect[]} render_effects
   */
  #b(t, n, r) {
    t.f ^= b;
    for (var i = t.first; i !== null; ) {
      var s = i.f, o = (s & (z | H)) !== 0, u = o && (s & b) !== 0, l = u || (s & F) !== 0 || this.#a.has(i);
      if (!l && i.fn !== null) {
        o ? i.f ^= b : (s & ye) !== 0 ? n.push(i) : be(i) && ((s & P) !== 0 && this.#u.add(i), ae(i));
        var f = i.first;
        if (f !== null) {
          i = f;
          continue;
        }
      }
      for (; i !== null; ) {
        var _ = i.next;
        if (_ !== null) {
          i = _;
          break;
        }
        i = i.parent;
      }
    }
  }
  #E() {
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
    for (const [r, i] of t.current)
      !this.previous.has(r) && t.previous.has(r) && this.previous.set(r, t.previous.get(r)), this.current.set(r, i);
    for (const [r, i] of t.async_deriveds) {
      const s = this.async_deriveds.get(r);
      s && i.promise.then(s.resolve).catch(s.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#f, t.#u);
    const n = (r) => {
      var i = r.reactions;
      if (i !== null)
        for (const u of i) {
          var s = u.f;
          if ((s & S) !== 0)
            n(
              /** @type {Derived} */
              u
            );
          else {
            var o = (
              /** @type {Effect} */
              u
            );
            s & (fe | P) && !this.async_deriveds.has(o) && (this.#u.delete(o), w(o, E), this.schedule(o));
          }
        }
    };
    for (const r of this.current.keys())
      n(r);
    this.oncommit(() => t.discard()), t.#d(), g = this, this.#g();
  }
  /**
   * @param {Effect[]} effects
   */
  #c(t) {
    for (var n = 0; n < t.length; n += 1)
      pt(t[n], this.#f, this.#u);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== m && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & $) === 0 && (this.current.set(t, [n, r]), C?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    g = this;
  }
  deactivate() {
    g = null, C = null;
  }
  flush() {
    try {
      qe = !0, g = this, this.#g();
    } finally {
      tt = 0, Ve = null, le = null, ke = null, qe = !1, g = null, C = null, J.clear();
    }
  }
  discard() {
    for (const t of this.#n) t(this);
    this.#n.clear();
    for (const t of this.async_deriveds.values())
      t.reject(_e);
    this.#d(), this.#i?.resolve();
  }
  /**
   * @param {Effect} effect
   */
  register_created_effect(t) {
    this.#_.push(t);
  }
  #m() {
    for (let a = Le; a !== null; a = a.#r) {
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
      var i = [...a.current.keys()].filter(
        (c) => !/** @type {[any, boolean]} */
        a.current.get(c)[1]
      );
      if (!(!a.#s || i.length === 0)) {
        var s = i.filter((c) => !this.current.has(c));
        if (s.length === 0)
          t && a.discard();
        else if (n.length > 0) {
          if (t)
            for (const c of this.#p)
              a.unskip_effect(c, (h) => {
                (h.f & (P | fe)) !== 0 ? a.schedule(h) : a.#c([h]);
              });
          a.activate();
          var o = /* @__PURE__ */ new Set(), u = /* @__PURE__ */ new Map();
          for (var l of n)
            bt(l, s, o, u);
          u = /* @__PURE__ */ new Map();
          var f = [...a.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (f.length > 0)
            for (const c of this.#_)
              (c.f & (N | F | Oe)) === 0 && We(c, f, u) && ((c.f & (fe | P)) !== 0 ? (w(c, E), a.schedule(c)) : a.#f.add(c));
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
    this.#o || (this.#o = !0, ue(() => {
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
    this.#h.add(t);
  }
  /** @param {(batch: Batch) => void} fn */
  ondiscard(t) {
    this.#n.add(t);
  }
  settled() {
    return (this.#i ??= ct()).promise;
  }
  static ensure() {
    if (g === null) {
      const t = g = new te();
      qe || ue(() => {
        t.#s || t.flush();
      });
    }
    return g;
  }
  apply() {
    {
      C = null;
      return;
    }
  }
  /**
   *
   * @param {Effect} effect
   */
  schedule(t) {
    if (Ve = t, t.b?.is_pending && (t.f & (ye | Fe | ht)) !== 0 && (t.f & ce) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (le !== null && n === p && (v === null || (v.f & S) === 0))
        return;
      if ((r & (H | z)) !== 0) {
        if ((r & b) === 0)
          return;
        n.f ^= b;
      }
    }
    this.#t.push(n);
  }
  #d() {
    if (this.linked) {
      var t = this.#v, n = this.#r;
      t === null ? Le = n : t.#r = n, n === null ? ie = t : n.#v = t, this.linked = !1;
    }
  }
}
function An() {
  try {
    nn();
  } catch (e) {
    V(e, Ve);
  }
}
let B = null;
function nt(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (N | F)) === 0 && be(r) && (B = /* @__PURE__ */ new Set(), ae(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Dt(r), B?.size > 0)) {
        J.clear();
        for (const i of B) {
          if ((i.f & (N | F)) !== 0) continue;
          const s = [i];
          let o = i.parent;
          for (; o !== null; )
            B.has(o) && (B.delete(o), s.push(o)), o = o.parent;
          for (let u = s.length - 1; u >= 0; u--) {
            const l = s[u];
            (l.f & (N | F)) === 0 && ae(l);
          }
        }
        B.clear();
      }
    }
    B = null;
  }
}
function bt(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const i of e.reactions) {
      const s = i.f;
      (s & S) !== 0 ? bt(
        /** @type {Derived} */
        i,
        t,
        n,
        r
      ) : (s & (fe | P)) !== 0 && (s & E) === 0 && We(i, t, r) && (w(i, E), Xe(
        /** @type {Effect} */
        i
      ));
    }
}
function We(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const i of e.deps) {
      if (Ae.call(t, i))
        return !0;
      if ((i.f & S) !== 0 && We(
        /** @type {Derived} */
        i,
        t,
        n
      ))
        return n.set(
          /** @type {Derived} */
          i,
          !0
        ), !0;
    }
  return n.set(e, !1), !1;
}
function Xe(e) {
  g.schedule(e);
}
function Et(e, t) {
  if (!((e.f & z) !== 0 && (e.f & b) !== 0)) {
    (e.f & E) !== 0 ? t.d.push(e) : (e.f & M) !== 0 && t.m.push(e), w(e, b);
    for (var n = e.first; n !== null; )
      Et(n, t), n = n.next;
  }
}
function St(e) {
  w(e, b);
  for (var t = e.first; t !== null; )
    St(t), t = t.next;
}
let De = /* @__PURE__ */ new Set();
const J = /* @__PURE__ */ new Map();
let kt = !1;
function Ie(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: _t,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function Y(e, t) {
  const n = Ie(e);
  return Yn(n), n;
}
function W(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!D || (v.f & Oe) !== 0) && me() && (v.f & (S | P | fe | Oe)) !== 0 && (I === null || !I.has(e)) && ln();
  let r = n ? de(t) : t;
  return Ne(e, r, ke);
}
function Ne(e, t, n = null) {
  if (!e.equals(t)) {
    J.set(e, ne ? t : e.v);
    var r = te.ensure();
    if (r.capture(e, t), (e.f & S) !== 0) {
      const i = (
        /** @type {Derived} */
        e
      );
      (e.f & E) !== 0 && Ke(i), C === null && Ge(i);
    }
    e.wv = jt(), xt(e, E, n), me() && p !== null && (p.f & b) !== 0 && (p.f & (z | H)) === 0 && (A === null ? Vn([e]) : A.push(e)), !r.is_fork && De.size > 0 && !kt && Rn();
  }
  return t;
}
function Rn() {
  kt = !1;
  for (const e of De) {
    (e.f & b) !== 0 && w(e, M);
    let t;
    try {
      t = be(e);
    } catch {
      t = !0;
    }
    t && ae(e);
  }
  De.clear();
}
function ge(e) {
  W(e, e.v + 1);
}
function xt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var i = me(), s = r.length, o = 0; o < s; o++) {
      var u = r[o], l = u.f;
      if (!(!i && u === p)) {
        var f = (l & E) === 0;
        if (f && w(u, t), (l & Oe) !== 0)
          De.add(
            /** @type {Effect} */
            u
          );
        else if ((l & S) !== 0) {
          var _ = (
            /** @type {Derived} */
            u
          );
          C?.delete(_), (l & ee) === 0 && (l & R && (p === null || (p.f & Pe) === 0) && (u.f |= ee), xt(_, M, n));
        } else if (f) {
          var a = (
            /** @type {Effect} */
            u
          );
          (l & P) !== 0 && B !== null && B.add(a), n !== null ? n.push(a) : Xe(a);
        }
      }
    }
}
function de(e) {
  if (typeof e != "object" || e === null || pe in e)
    return e;
  const t = at(e);
  if (t !== Ht && t !== Gt)
    return e;
  var n = /* @__PURE__ */ new Map(), r = Ut(e), i = /* @__PURE__ */ Y(0), s = Q, o = (u) => {
    if (Q === s)
      return u();
    var l = v, f = Q;
    O(null), st(s);
    var _ = u();
    return O(l), st(f), _;
  };
  return r && n.set("length", /* @__PURE__ */ Y(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(u, l, f) {
        (!("value" in f) || f.configurable === !1 || f.enumerable === !1 || f.writable === !1) && rn();
        var _ = n.get(l);
        return _ === void 0 ? o(() => {
          var a = /* @__PURE__ */ Y(f.value);
          return n.set(l, a), a;
        }) : W(_, f.value, !0), !0;
      },
      deleteProperty(u, l) {
        var f = n.get(l);
        if (f === void 0) {
          if (l in u) {
            const _ = o(() => /* @__PURE__ */ Y(m));
            n.set(l, _), ge(i);
          }
        } else
          W(f, m), ge(i);
        return !0;
      },
      get(u, l, f) {
        if (l === pe)
          return e;
        var _ = n.get(l), a = l in u;
        if (_ === void 0 && (!a || ve(u, l)?.writable) && (_ = o(() => {
          var h = de(a ? u[l] : m), d = /* @__PURE__ */ Y(h);
          return d;
        }), n.set(l, _)), _ !== void 0) {
          var c = Z(_);
          return c === m ? void 0 : c;
        }
        return Reflect.get(u, l, f);
      },
      getOwnPropertyDescriptor(u, l) {
        var f = Reflect.getOwnPropertyDescriptor(u, l);
        if (f && "value" in f) {
          var _ = n.get(l);
          _ && (f.value = Z(_));
        } else if (f === void 0) {
          var a = n.get(l), c = a?.v;
          if (a !== void 0 && c !== m)
            return {
              enumerable: !0,
              configurable: !0,
              value: c,
              writable: !0
            };
        }
        return f;
      },
      has(u, l) {
        if (l === pe)
          return !0;
        var f = n.get(l), _ = f !== void 0 && f.v !== m || Reflect.has(u, l);
        if (f !== void 0 || p !== null && (!_ || ve(u, l)?.writable)) {
          f === void 0 && (f = o(() => {
            var c = _ ? de(u[l]) : m, h = /* @__PURE__ */ Y(c);
            return h;
          }), n.set(l, f));
          var a = Z(f);
          if (a === m)
            return !1;
        }
        return _;
      },
      set(u, l, f, _) {
        var a = n.get(l), c = l in u;
        if (r && l === "length")
          for (var h = f; h < /** @type {Source<number>} */
          a.v; h += 1) {
            var d = n.get(h + "");
            d !== void 0 ? W(d, m) : h in u && (d = o(() => /* @__PURE__ */ Y(m)), n.set(h + "", d));
          }
        if (a === void 0)
          (!c || ve(u, l)?.writable) && (a = o(() => /* @__PURE__ */ Y(void 0)), W(a, de(f)), n.set(l, a));
        else {
          c = a.v !== m;
          var y = o(() => de(f));
          W(a, y);
        }
        var U = Reflect.getOwnPropertyDescriptor(u, l);
        if (U?.set && U.set.call(_, f), !c) {
          if (r && typeof l == "string") {
            var q = (
              /** @type {Source<number>} */
              n.get("length")
            ), re = Number(l);
            Number.isInteger(re) && re >= q.v && W(q, re + 1);
          }
          ge(i);
        }
        return !0;
      },
      ownKeys(u) {
        Z(i);
        var l = Reflect.ownKeys(u).filter((a) => {
          var c = n.get(a);
          return c === void 0 || c.v !== m;
        });
        for (var [f, _] of n)
          _.v !== m && !(f in u) && l.push(f);
        return l;
      },
      setPrototypeOf() {
        sn();
      }
    }
  );
}
var rt, On, Tt, At;
function Pn() {
  if (rt === void 0) {
    rt = window, On = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    Tt = ve(t, "firstChild").get, At = ve(t, "nextSibling").get, Qe(e) && (e[Qt] = void 0, e[Jt] = null, e[en] = void 0, e.__e = void 0), Qe(n) && (n[Ue] = void 0);
  }
}
function Rt(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function Cn(e) {
  return (
    /** @type {TemplateNode | null} */
    Tt.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function Ot(e) {
  return (
    /** @type {TemplateNode | null} */
    At.call(e)
  );
}
function rr(e, t) {
  return /* @__PURE__ */ Cn(e);
}
function ir(e, t, n) {
  return (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  );
}
function Pt(e) {
  var t = v, n = p;
  O(null), L(null);
  try {
    return e();
  } finally {
    O(t), L(n);
  }
}
function Dn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function G(e, t) {
  var n = p;
  n !== null && (n.f & F) !== 0 && (e |= F);
  var r = {
    ctx: T,
    deps: null,
    nodes: null,
    f: e | E | R,
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
  var i = r;
  if ((e & ye) !== 0)
    le !== null ? le.push(r) : te.ensure().schedule(r);
  else if (t !== null) {
    try {
      ae(r);
    } catch (o) {
      throw j(r), o;
    }
    i.deps === null && i.teardown === null && i.nodes === null && i.first === i.last && // either `null`, or a singular child
    (i.f & he) === 0 && (i = i.first, (e & P) !== 0 && (e & Re) !== 0 && i !== null && (i.f |= Re));
  }
  if (i !== null && (i.parent = n, n !== null && Dn(i, n), v !== null && (v.f & S) !== 0 && (e & H) === 0)) {
    var s = (
      /** @type {Derived} */
      v
    );
    (s.effects ??= []).push(i);
  }
  return r;
}
function Ze() {
  return v !== null && !D;
}
function Nn(e) {
  const t = G(Fe, null);
  return w(t, b), t.teardown = e, t;
}
function Fn(e) {
  return G(ye | Zt, e);
}
function jn(e) {
  te.ensure();
  const t = G(H | he, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? xe(t, () => {
      j(t), r(void 0);
    }) : (j(t), r(void 0));
  });
}
function In(e) {
  return G(fe | he, e);
}
function Mn(e, t = 0) {
  return G(Fe | t, e);
}
function sr(e, t = [], n = [], r = []) {
  mn(r, t, n, (i) => {
    G(Fe, () => {
      e(...i.map(Z));
    });
  });
}
function Ln(e, t = 0) {
  var n = G(P | t, e);
  return n;
}
function K(e) {
  return G(z | he, e);
}
function Ct(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = ne, r = v;
    it(!0), O(null);
    try {
      t.call(null);
    } finally {
      it(n), O(r);
    }
  }
}
function Je(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const i = n.ac;
    i !== null && Pt(() => {
      i.abort(je);
    });
    var r = n.next;
    (n.f & H) !== 0 ? n.parent = null : j(n, t), n = r;
  }
}
function qn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & z) === 0 && j(t), t = n;
  }
}
function j(e, t = !0) {
  var n = !1;
  (t || (e.f & Xt) !== 0) && e.nodes !== null && e.nodes.end !== null && (Bn(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= et, Je(e, t && !n), we(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const s of r)
      s.stop();
  Ct(e), e.f ^= et, e.f |= N;
  var i = e.parent;
  i !== null && i.first !== null && Dt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function Bn(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ Ot(e);
    e.remove(), e = n;
  }
}
function Dt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function xe(e, t, n = !0) {
  var r = [];
  Nt(e, r, !0);
  var i = () => {
    n && j(e), t && t();
  }, s = r.length;
  if (s > 0) {
    var o = () => --s || i();
    for (var u of r)
      u.out(o);
  } else
    i();
}
function Nt(e, t, n) {
  if ((e.f & F) === 0) {
    e.f ^= F;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const u of r)
        (u.is_global || n) && t.push(u);
    for (var i = e.first; i !== null; ) {
      var s = i.next;
      if ((i.f & H) === 0) {
        var o = (i.f & Re) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (i.f & z) !== 0 && (e.f & P) !== 0;
        Nt(i, t, o ? n : !1);
      }
      i = s;
    }
  }
}
function Un(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var i = n === r ? null : /* @__PURE__ */ Ot(n);
      t.append(n), n = i;
    }
}
let Te = !1, ne = !1;
function it(e) {
  ne = e;
}
let v = null, D = !1;
function O(e) {
  v = e;
}
let p = null;
function L(e) {
  p = e;
}
let I = null;
function Yn(e) {
  v !== null && (I ??= /* @__PURE__ */ new Set()).add(e);
}
let k = null, x = 0, A = null;
function Vn(e) {
  A = e;
}
let Ft = 1, X = 0, Q = X;
function st(e) {
  Q = e;
}
function jt() {
  return ++Ft;
}
function be(e) {
  var t = e.f;
  if ((t & E) !== 0)
    return !0;
  if (t & S && (e.f &= ~ee), (t & M) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, i = 0; i < r; i++) {
      var s = n[i];
      if (be(
        /** @type {Derived} */
        s
      ) && wt(
        /** @type {Derived} */
        s
      ), s.wv > e.wv)
        return !0;
    }
    (t & R) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    C === null && w(e, b);
  }
  return !1;
}
function It(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(I !== null && I.has(e)))
    for (var i = 0; i < r.length; i++) {
      var s = r[i];
      (s.f & S) !== 0 ? It(
        /** @type {Derived} */
        s,
        t,
        !1
      ) : t === s && (n ? w(s, E) : (s.f & b) !== 0 && w(s, M), Xe(
        /** @type {Effect} */
        s
      ));
    }
}
function Mt(e) {
  var t = k, n = x, r = A, i = v, s = I, o = T, u = D, l = Q, f = e.f;
  k = /** @type {null | Value[]} */
  null, x = 0, A = null, v = (f & (z | H)) === 0 ? e : null, I = null, oe(e.ctx), D = !1, Q = ++X, e.ac !== null && (Pt(() => {
    e.ac.abort(je);
  }), e.ac = null);
  try {
    e.f |= Pe;
    var _ = (
      /** @type {Function} */
      e.fn
    ), a = _();
    e.f |= ce;
    var c = e.deps, h = g?.is_fork;
    if (k !== null) {
      var d;
      if (h || we(e, x), c !== null && x > 0)
        for (c.length = x + k.length, d = 0; d < k.length; d++)
          c[x + d] = k[d];
      else
        e.deps = c = k;
      if (Ze() && (e.f & R) !== 0)
        for (d = x; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && x < c.length && (we(e, x), c.length = x);
    if (me() && A !== null && !D && c !== null && (e.f & (S | M | E)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      A.length; d++)
        It(
          A[d],
          /** @type {Effect} */
          e
        );
    if (i !== null && i !== e) {
      if (X++, i.deps !== null)
        for (let y = 0; y < n; y += 1)
          i.deps[y].rv = X;
      if (t !== null)
        for (const y of t)
          y.rv = X;
      A !== null && (r === null ? r = A : r.push(.../** @type {Source[]} */
      A));
    }
    return (e.f & $) !== 0 && (e.f ^= $), a;
  } catch (y) {
    return dt(y);
  } finally {
    e.f ^= Pe, k = t, x = n, A = r, v = i, I = s, oe(o), D = u, Q = l;
  }
}
function $n(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = Yt.call(n, e);
    if (r !== -1) {
      var i = n.length - 1;
      i === 0 ? n = t.reactions = null : (n[r] = n[i], n.pop());
    }
  }
  if (n === null && (t.f & S) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (k === null || !Ae.call(k, t))) {
    var s = (
      /** @type {Derived} */
      t
    );
    (s.f & R) !== 0 && (s.f ^= R, s.f &= ~ee), s.v !== m && Ge(s), xn(s), we(s, 0);
  }
}
function we(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      $n(e, n[r]);
}
function ae(e) {
  var t = e.f;
  if ((t & N) === 0) {
    w(e, b);
    var n = p, r = Te;
    p = e, Te = !0;
    try {
      (t & (P | ht)) !== 0 ? qn(e) : Je(e), Ct(e);
      var i = Mt(e);
      e.teardown = typeof i == "function" ? i : null, e.wv = Ft;
      var s;
      ot && Bt && (e.f & E) !== 0 && e.deps;
    } finally {
      Te = r, p = n;
    }
  }
}
function Z(e) {
  var t = e.f, n = (t & S) !== 0;
  if (v !== null && !D) {
    var r = p !== null && (p.f & N) !== 0;
    if (!r && (I === null || !I.has(e))) {
      var i = v.deps;
      if ((v.f & Pe) !== 0)
        e.rv < X && (e.rv = X, k === null && i !== null && i[x] === e ? x++ : k === null ? k = [e] : k.push(e));
      else {
        v.deps ??= [], Ae.call(v.deps, e) || v.deps.push(e);
        var s = e.reactions;
        s === null ? e.reactions = [v] : Ae.call(s, v) || s.push(v);
      }
    }
  }
  if (ne && J.has(e))
    return J.get(e);
  if (n) {
    var o = (
      /** @type {Derived} */
      e
    );
    if (ne) {
      var u = o.v;
      return ((o.f & b) === 0 && o.reactions !== null || qt(o)) && (u = Ke(o)), J.set(o, u), u;
    }
    var l = (o.f & R) === 0 && !D && v !== null && (Te || (v.f & R) !== 0), f = (o.f & ce) === 0;
    be(o) && (l && (o.f |= R), wt(o)), l && !f && (mt(o), Lt(o));
  }
  if (C?.has(e))
    return C.get(e);
  if ((e.f & $) !== 0)
    throw e.v;
  return e.v;
}
function Lt(e) {
  if (e.f |= R, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & S) !== 0 && (t.f & R) === 0 && (mt(
        /** @type {Derived} */
        t
      ), Lt(
        /** @type {Derived} */
        t
      ));
}
function qt(e) {
  if (e.v === m) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (J.has(t) || (t.f & S) !== 0 && qt(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function zn(e) {
  var t = D;
  try {
    return D = !0, e();
  } finally {
    D = t;
  }
}
function lr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (pe in e)
      $e(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && pe in n && $e(n);
      }
  }
}
function $e(e, t = /* @__PURE__ */ new Set()) {
  if (typeof e == "object" && e !== null && // We don't want to traverse DOM elements
  !(e instanceof EventTarget) && !t.has(e)) {
    t.add(e), e instanceof Date && e.getTime();
    for (let r in e)
      try {
        $e(e[r], t);
      } catch {
      }
    const n = at(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = zt(n);
      for (let i in r) {
        const s = r[i].get;
        if (s)
          try {
            s.call(e);
          } catch {
          }
      }
    }
  }
}
const Ee = /* @__PURE__ */ Symbol("events"), Hn = /* @__PURE__ */ new Set(), lt = /* @__PURE__ */ new Set();
let ft = null;
function ut(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, i = e.composedPath?.() || [], s = (
    /** @type {null | Element} */
    i[0] || e.target
  );
  ft = e;
  var o = 0, u = ft === e && e[Ee];
  if (u) {
    var l = i.indexOf(u);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[Ee] = t;
      return;
    }
    var f = i.indexOf(t);
    if (f === -1)
      return;
    l <= f && (o = l);
  }
  if (s = /** @type {Element} */
  i[o] || e.target, s !== t) {
    $t(e, "currentTarget", {
      configurable: !0,
      get() {
        return s || n;
      }
    });
    var _ = v, a = p;
    O(null), L(null);
    try {
      for (var c, h = []; s !== null && s !== t; ) {
        try {
          var d = s[Ee]?.[r];
          d != null && (!/** @type {any} */
          s.disabled || // DOM could've been updated already by the time this is reached, so we check this as well
          // -> the target could not have been disabled because it emits the event in the first place
          e.target === s) && d.call(s, e);
        } catch (y) {
          c ? h.push(y) : c = y;
        }
        if (e.cancelBubble) break;
        o++, s = o < i.length ? (
          /** @type {Element} */
          i[o]
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
      e[Ee] = t, delete e.currentTarget, O(_), L(a);
    }
  }
}
const Gn = ["touchstart", "touchmove"];
function Kn(e) {
  return Gn.includes(e);
}
function fr(e, t) {
  var n = t == null ? "" : typeof t == "object" ? `${t}` : t;
  n !== /** @type {any} */
  (e[Ue] ??= e.nodeValue) && (e[Ue] = n, e.nodeValue = `${n}`);
}
function ur(e, t) {
  return Wn(e, t);
}
const Se = /* @__PURE__ */ new Map();
function Wn(e, { target: t, anchor: n, props: r = {}, events: i, context: s, intro: o = !0, transformError: u }) {
  Pn();
  var l = void 0, f = jn(() => {
    var _ = n ?? t.appendChild(Rt());
    yn(
      /** @type {TemplateNode} */
      _,
      {
        pending: () => {
        }
      },
      (h) => {
        hn({});
        var d = (
          /** @type {ComponentContext} */
          T
        );
        s && (d.c = s), i && (r.$$events = i), l = e(h, r) || {}, _n();
      },
      u
    );
    var a = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var y = h[d];
        if (!a.has(y)) {
          a.add(y);
          var U = Kn(y);
          for (const Me of [t, document]) {
            var q = Se.get(Me);
            q === void 0 && (q = /* @__PURE__ */ new Map(), Se.set(Me, q));
            var re = q.get(y);
            re === void 0 ? (Me.addEventListener(y, ut, { passive: U }), q.set(y, 1)) : q.set(y, re + 1);
          }
        }
      }
    };
    return c(Vt(Hn)), lt.add(c), () => {
      for (var h of a)
        for (const U of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            Se.get(U)
          ), y = (
            /** @type {number} */
            d.get(h)
          );
          --y == 0 ? (U.removeEventListener(h, ut), d.delete(h), d.size === 0 && Se.delete(U)) : d.set(h, y);
        }
      lt.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return ze.set(l, f), l;
}
let ze = /* @__PURE__ */ new WeakMap();
function or(e, t) {
  const n = ze.get(e);
  return n ? (ze.delete(e), n(t)) : Promise.resolve();
}
export {
  N as D,
  tr as L,
  Jn as P,
  pe as S,
  er as T,
  p as a,
  ve as b,
  ir as c,
  Z as d,
  Xn as e,
  Zn as f,
  Cn as g,
  Qn as h,
  On as i,
  Sn as j,
  ne as k,
  He as l,
  lr as m,
  fr as n,
  rr as o,
  de as p,
  ur as q,
  or as r,
  W as s,
  sr as t,
  zn as u
};
