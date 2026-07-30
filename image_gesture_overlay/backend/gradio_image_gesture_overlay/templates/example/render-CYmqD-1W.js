let Ge = !1, Bt = !1;
function Xn() {
  Ge = !0;
}
const Zn = 2, Jn = 4, Qn = 8, er = 2, m = /* @__PURE__ */ Symbol("uninitialized"), at = !1;
var Ut = Array.isArray, Yt = Array.prototype.indexOf, Re = Array.prototype.includes, Vt = Array.from, $t = Object.defineProperty, ve = Object.getOwnPropertyDescriptor, zt = Object.getOwnPropertyDescriptors, Ht = Object.prototype, Gt = Array.prototype, ct = Object.getPrototypeOf, et = Object.isExtensible;
const Kt = () => {
};
function Wt(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function ht() {
  var e, t, n = new Promise((r, i) => {
    e = r, t = i;
  });
  return { promise: n, resolve: e, reject: t };
}
const S = 2, ye = 4, je = 8, _t = 1 << 24, P = 16, U = 32, Y = 64, Ue = 128, R = 512, E = 1024, b = 2048, N = 4096, j = 8192, F = 16384, ce = 32768, tt = 1 << 25, Oe = 65536, Pe = 1 << 17, Xt = 1 << 18, he = 1 << 19, Zt = 1 << 20, te = 65536, Ce = 1 << 21, oe = 1 << 22, H = 1 << 23, pe = /* @__PURE__ */ Symbol("$state"), tr = /* @__PURE__ */ Symbol("legacy props"), Jt = /* @__PURE__ */ Symbol("attributes"), Qt = /* @__PURE__ */ Symbol("class"), en = /* @__PURE__ */ Symbol("style"), Ye = /* @__PURE__ */ Symbol("text"), me = new class extends Error {
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
function on() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function un() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function dt(e) {
  return e === this.v;
}
function an(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function cn(e) {
  return !an(e, this.v);
}
let T = null;
function ue(e) {
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
    l: Ge && !t ? { s: null, u: null, $: [] } : null
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
function be() {
  return !Ge || T !== null && T.l === null;
}
let le = [];
function dn() {
  var e = le;
  le = [], Wt(e);
}
function X(e) {
  if (le.length === 0) {
    var t = le;
    queueMicrotask(() => {
      t === le && dn();
    });
  }
  le.push(e);
}
function vt(e) {
  var t = p;
  if (t === null)
    return v.f |= H, e;
  if ((t.f & ce) === 0 && (t.f & ye) === 0)
    throw e;
  z(e, t);
}
function z(e, t) {
  if (!(t !== null && (t.f & F) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & Ue) !== 0) {
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
function Ke(e) {
  (e.f & R) !== 0 || e.deps === null ? w(e, E) : w(e, N);
}
function pt(e) {
  if (e !== null)
    for (const t of e)
      (t.f & S) === 0 || (t.f & te) === 0 || (t.f ^= te, pt(
        /** @type {Derived} */
        t.deps
      ));
}
function gt(e, t, n) {
  (e.f & b) !== 0 ? t.add(e) : (e.f & N) !== 0 && n.add(e), pt(e.deps), w(e, E);
}
function Ie(e) {
  var t = v, n = p;
  O(null), L(null);
  try {
    return e();
  } finally {
    O(t), L(n);
  }
}
function pn(e) {
  let t = 0, n = Me(0), r;
  return () => {
    Je() && (J(n), Mn(() => (t === 0 && (r = zn(() => e(() => ge(n)))), t += 1, () => {
      X(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, ge(n));
      });
    })));
  };
}
var gn = Oe | he;
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
  #c;
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
  #d = 0;
  #f = 0;
  #o = !1;
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
  #u = null;
  #y = pn(() => (this.#u = Me(this.#d), () => {
    this.#u = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(t, n, r, i) {
    this.#s = t, this.#r = n, this.#c = (s) => {
      var u = (
        /** @type {Effect} */
        p
      );
      u.b = this, u.f |= Ue, r(s);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = i ?? this.parent?.transform_error ?? ((s) => s), this.#n = Ln(() => {
      this.#h();
    }, gn);
  }
  #g() {
    try {
      this.#l = K(() => this.#c(this.#s));
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #b(t) {
    const n = this.#r.failed, { reset: r, invoke_onerror: i } = this.#w(t);
    X(i), n && (this.#i = K(() => {
      n(
        this.#s,
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
    const i = () => {
      if (n) {
        un();
        return;
      }
      n = !0, r && fn(), this.#i !== null && Te(this.#i, () => {
        this.#i = null;
      }), this.#_(() => {
        this.#h();
      });
    };
    return { reset: i, invoke_onerror: () => {
      try {
        r = !0, this.#r.onerror?.(t, i), r = !1;
      } catch (u) {
        z(u, this.#n && this.#n.parent);
      }
    } };
  }
  #E() {
    const t = this.#r.pending;
    t && (this.is_pending = !0, this.#e = K(() => t(this.#s)), X(() => {
      var n = this.#t = document.createDocumentFragment(), r = Ot();
      n.append(r), this.#l = this.#_(() => K(() => this.#c(r))), this.#f === 0 && (this.#s.before(n), this.#t = null, Te(
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
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#d = 0, this.#l = K(() => {
        this.#c(this.#s);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        Un(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = K(() => n(this.#s));
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
    gt(t, this.#a, this.#p);
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
    var n = p, r = v, i = T;
    L(this.#n), O(this.#n), ue(this.#n.ctx);
    try {
      return ne.ensure(), t();
    } catch (s) {
      return vt(s), null;
    } finally {
      L(n), O(r), ue(i);
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
    this.#f += t, this.#f === 0 && (this.#m(n), this.#e && Te(this.#e, () => {
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
    this.#S(t, n), this.#d += t, !(!this.#u || this.#o) && (this.#o = !0, X(() => {
      this.#o = !1, this.#u && Fe(this.#u, this.#d);
    }));
  }
  get_effect_pending() {
    return this.#y(), J(
      /** @type {Source<number>} */
      this.#u
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    g?.is_fork ? (this.#l && g.skip_effect(this.#l), this.#e && g.skip_effect(this.#e), this.#i && g.skip_effect(this.#i), g.oncommit(() => {
      this.#k(t);
    })) : this.#k(t);
  }
  /**
   * @param {unknown} error
   */
  #k(t) {
    this.#l && (I(this.#l), this.#l = null), this.#e && (I(this.#e), this.#e = null), this.#i && (I(this.#i), this.#i = null);
    let n = this.#r.failed;
    const r = (i) => {
      const { reset: s, invoke_onerror: u } = this.#w(i);
      u(), n && (this.#i = this.#_(() => {
        try {
          return K(() => {
            var f = (
              /** @type {Effect} */
              p
            );
            f.b = this, f.f |= Ue, n(
              this.#s,
              () => i,
              () => s
            );
          });
        } catch (f) {
          return z(
            f,
            /** @type {Effect} */
            this.#n.parent
          ), null;
        }
      }));
    };
    X(() => {
      var i;
      try {
        i = this.transform_error(t);
      } catch (s) {
        z(s, this.#n && this.#n.parent);
        return;
      }
      i !== null && typeof i == "object" && typeof /** @type {any} */
      i.then == "function" ? i.then(
        r,
        /** @param {unknown} e */
        (s) => z(s, this.#n && this.#n.parent)
      ) : r(i);
    });
  }
}
function mn(e, t, n, r) {
  const i = be() ? wt : Sn;
  var s = e.filter((h) => !h.settled), u = t.map(i);
  if (n.length === 0 && s.length === 0) {
    r(u);
    return;
  }
  var f = (
    /** @type {Effect} */
    p
  ), l = bn(), o = s.length === 1 ? s[0].promise : s.length > 1 ? Promise.all(s.map((h) => h.promise)) : null;
  function _(h) {
    if ((f.f & F) === 0) {
      l();
      try {
        r([...u, ...h]);
      } catch (d) {
        z(d, f);
      }
      De();
    }
  }
  var a = yt();
  if (n.length === 0) {
    o.then(() => _([])).finally(a);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ En(h))).then(_).catch((h) => z(h, f)).finally(a);
  }
  o ? o.then(() => {
    l(), c(), De();
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
    L(e), O(t), ue(n), s && (e.f & F) === 0 && (r?.activate(), r?.apply());
  };
}
function De(e = !0) {
  L(null), O(null), ue(null), e && g?.deactivate();
}
function yt() {
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
function wt(e) {
  var t = S | b;
  return p !== null && (p.f |= he), {
    ctx: T,
    deps: null,
    effects: null,
    equals: dt,
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
  ), s = Me(
    /** @type {V} */
    m
  ), u = !v, f = /* @__PURE__ */ new Set();
  return In(() => {
    var l = (
      /** @type {Effect} */
      p
    ), o = ht();
    i = o.promise;
    try {
      Promise.resolve(e()).then(o.resolve, (h) => {
        h !== me && o.reject(h);
      }).finally(De);
    } catch (h) {
      o.reject(h), De();
    }
    var _ = (
      /** @type {Batch} */
      g
    );
    if (u) {
      if ((l.f & ce) !== 0)
        var a = yt();
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
      a?.(), f.delete(o), d !== _e && (_.activate(), d ? (s.f |= H, Fe(s, d)) : ((s.f & H) !== 0 && (s.f ^= H), Fe(s, h)), _.deactivate());
    };
    o.promise.then(c, (h) => c(null, h || "unknown"));
  }), Nn(() => {
    for (const l of f)
      l.reject(_e);
  }), new Promise((l) => {
    function o(_) {
      function a() {
        _ === i ? l(s) : o(i);
      }
      _.then(a, a);
    }
    o(i);
  });
}
// @__NO_SIDE_EFFECTS__
function Sn(e) {
  const t = /* @__PURE__ */ wt(e);
  return t.equals = cn, t;
}
function kn(e) {
  var t = e.effects;
  if (t !== null) {
    e.effects = null;
    for (var n = 0; n < t.length; n += 1)
      I(
        /** @type {Effect} */
        t[n]
      );
  }
}
function We(e) {
  var t, n = p, r = e.parent;
  if (!re && r !== null && e.v !== m && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (F | j)) !== 0)
    return on(), e.v;
  L(r);
  try {
    e.f &= ~te, kn(e), t = Mt(e);
  } finally {
    L(n);
  }
  return t;
}
function mt(e) {
  var t = We(e);
  if (!e.equals(t) && (e.wv = jt(), (!g?.is_fork || e.deps === null) && (g !== null ? (g.capture(e, t, !0), Ve?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, E);
    return;
  }
  re || (C !== null ? (Je() || g?.is_fork) && C.set(e, t) : Ke(e));
}
function xn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac !== null && Ie(() => {
        t.ac.abort(me), t.ac = null;
      }), t.fn !== null && (t.teardown = Kt), we(t, 0), Qe(t));
}
function bt(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && ae(t);
}
let qe = null, se = null, g = null, Ve = null, C = null, $e = null, Be = !1, fe = null, xe = null;
var nt = 0;
let Tn = 1;
class ne {
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
  #o = /* @__PURE__ */ new Set();
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
  #u = !1;
  constructor() {
    se === null ? qe = se = this : (se.#r = this, this.#v = se), se = this;
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
        w(i, b), n(i);
      for (i of r.m)
        w(i, N), n(i);
    }
    this.#p.add(t);
  }
  #g() {
    this.#s = !0, nt++ > 1e3 && (this.#_(), An());
    for (const l of this.#f)
      this.#o.delete(l), w(l, b), this.schedule(l);
    for (const l of this.#o)
      w(l, N), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = fe = [], r = [], i = xe = [];
    for (const l of t)
      try {
        this.#b(l, n, r);
      } catch (o) {
        throw kt(l), this.#y() || this.discard(), o;
      }
    if (g = null, i.length > 0) {
      var s = ne.ensure();
      for (const l of i)
        s.schedule(l);
    }
    if (fe = null, xe = null, this.#y()) {
      this.#h(r), this.#h(n);
      for (const [l, o] of this.#a)
        St(l, o);
      i.length > 0 && /** @type {unknown} */
      g.#g();
      return;
    }
    const u = this.#w();
    if (u) {
      this.#h(r), this.#h(n), u.#E(this);
      return;
    }
    this.#f.clear(), this.#o.clear();
    for (const l of this.#c) l(this);
    this.#c.clear(), Ve = this, rt(r), rt(n), Ve = null, this.#i?.resolve();
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
    f !== null && f.#g();
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
    for (var i = t.first; i !== null; ) {
      var s = i.f, u = (s & (U | Y)) !== 0, f = u && (s & E) !== 0, l = f || (s & j) !== 0 || this.#a.has(i);
      if (!l && i.fn !== null) {
        u ? i.f ^= E : (s & ye) !== 0 ? n.push(i) : Ee(i) && ((s & P) !== 0 && this.#o.add(i), ae(i));
        var o = i.first;
        if (o !== null) {
          i = o;
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
    for (const [r, i] of t.current)
      !this.previous.has(r) && t.previous.has(r) && this.previous.set(r, t.previous.get(r)), this.current.set(r, i);
    for (const [r, i] of t.async_deriveds) {
      const s = this.async_deriveds.get(r);
      s && i.promise.then(s.resolve).catch(s.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#f, t.#o);
    const n = (r) => {
      var i = r.reactions;
      if (i !== null && !((r.f & S) !== 0 && (r.f & (b | N)) === 0))
        for (const f of i) {
          var s = f.f;
          if ((s & S) !== 0)
            n(
              /** @type {Derived} */
              f
            );
          else {
            var u = (
              /** @type {Effect} */
              f
            );
            s & (oe | P) && !this.async_deriveds.has(u) && (this.#o.delete(u), w(u, b), this.schedule(u));
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
      gt(t[n], this.#f, this.#o);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== m && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & H) === 0 && (this.current.set(t, [n, r]), C?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    g = this;
  }
  deactivate() {
    g = null, C = null;
  }
  flush() {
    try {
      Be = !0, g = this, this.#g();
    } finally {
      nt = 0, $e = null, fe = null, xe = null, Be = !1, g = null, C = null, Q.clear();
    }
  }
  discard() {
    for (const t of this.#n) t(this);
    this.#n.clear();
    for (const t of this.async_deriveds.values())
      t.reject(_e);
    this.#_(), this.#i?.resolve();
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
                (h.f & (P | oe)) !== 0 ? a.schedule(h) : a.#h([h]);
              });
          a.activate();
          var u = /* @__PURE__ */ new Set(), f = /* @__PURE__ */ new Map();
          for (var l of n)
            Et(l, s, u, f);
          f = /* @__PURE__ */ new Map();
          var o = [...a.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (o.length > 0)
            for (const c of this.#d)
              (c.f & (F | j | Pe)) === 0 && Xe(c, o, f) && ((c.f & (oe | P)) !== 0 ? (w(c, b), a.schedule(c)) : a.#f.add(c));
          if (a.#t.length > 0 && !a.#u) {
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
    this.#u || (this.#u = !0, X(() => {
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
      this.#o.add(r);
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
    return (this.#i ??= ht()).promise;
  }
  static ensure() {
    if (g === null) {
      const t = g = new ne();
      Be || X(() => {
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
    if ($e = t, t.b?.is_pending && (t.f & (ye | je | _t)) !== 0 && (t.f & ce) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (fe !== null && n === p && (v === null || (v.f & S) === 0))
        return;
      if ((r & (Y | U)) !== 0) {
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
      t === null ? qe = n : t.#r = n, n === null ? se = t : n.#v = t, this.linked = !1;
    }
  }
}
function An() {
  try {
    nn();
  } catch (e) {
    z(e, $e);
  }
}
let B = null;
function rt(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (F | j)) === 0 && Ee(r) && (B = /* @__PURE__ */ new Set(), ae(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Dt(r), B?.size > 0)) {
        Q.clear();
        for (const i of B) {
          if ((i.f & (F | j)) !== 0) continue;
          const s = [i];
          let u = i.parent;
          for (; u !== null; )
            B.has(u) && (B.delete(u), s.push(u)), u = u.parent;
          for (let f = s.length - 1; f >= 0; f--) {
            const l = s[f];
            (l.f & (F | j)) === 0 && ae(l);
          }
        }
        B.clear();
      }
    }
    B = null;
  }
}
function Et(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const i of e.reactions) {
      const s = i.f;
      (s & S) !== 0 ? Et(
        /** @type {Derived} */
        i,
        t,
        n,
        r
      ) : (s & (oe | P)) !== 0 && (s & b) === 0 && Xe(i, t, r) && (w(i, b), Ze(
        /** @type {Effect} */
        i
      ));
    }
}
function Xe(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const i of e.deps) {
      if (Re.call(t, i))
        return !0;
      if ((i.f & S) !== 0 && Xe(
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
function Ze(e) {
  g.schedule(e);
}
function St(e, t) {
  if (!((e.f & U) !== 0 && (e.f & E) !== 0)) {
    (e.f & b) !== 0 ? t.d.push(e) : (e.f & N) !== 0 && t.m.push(e), w(e, E);
    for (var n = e.first; n !== null; )
      St(n, t), n = n.next;
  }
}
function kt(e) {
  w(e, E);
  for (var t = e.first; t !== null; )
    kt(t), t = t.next;
}
let Ne = /* @__PURE__ */ new Set();
const Q = /* @__PURE__ */ new Map();
let xt = !1;
function Me(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: dt,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function $(e, t) {
  const n = Me(e);
  return Yn(n), n;
}
function W(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!D || (v.f & Pe) !== 0) && be() && (v.f & (S | P | oe | Pe)) !== 0 && (M === null || !M.has(e)) && ln();
  let r = n ? de(t) : t;
  return Fe(e, r, xe);
}
function Fe(e, t, n = null) {
  if (!e.equals(t)) {
    Q.set(e, re ? t : e.v);
    var r = ne.ensure();
    if (r.capture(e, t), (e.f & S) !== 0) {
      const i = (
        /** @type {Derived} */
        e
      );
      (e.f & b) !== 0 && We(i), C === null && Ke(i);
    }
    e.wv = jt(), Tt(e, b, n), be() && p !== null && (p.f & E) !== 0 && (p.f & (U | Y)) === 0 && (A === null ? Vn([e]) : A.push(e)), !r.is_fork && Ne.size > 0 && !xt && Rn();
  }
  return t;
}
function Rn() {
  xt = !1;
  for (const e of Ne) {
    (e.f & E) !== 0 && w(e, N);
    let t;
    try {
      t = Ee(e);
    } catch {
      t = !0;
    }
    t && ae(e);
  }
  Ne.clear();
}
function ge(e) {
  W(e, e.v + 1);
}
function Tt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var i = be(), s = r.length, u = 0; u < s; u++) {
      var f = r[u], l = f.f;
      if (!(!i && f === p)) {
        var o = (l & b) === 0;
        if (o && w(f, t), (l & Pe) !== 0)
          Ne.add(
            /** @type {Effect} */
            f
          );
        else if ((l & S) !== 0) {
          var _ = (
            /** @type {Derived} */
            f
          );
          C?.delete(_), (l & te) === 0 && (l & R && (p === null || (p.f & Ce) === 0) && (f.f |= te), Tt(_, N, n));
        } else if (o) {
          var a = (
            /** @type {Effect} */
            f
          );
          (l & P) !== 0 && B !== null && B.add(a), n !== null ? n.push(a) : Ze(a);
        }
      }
    }
}
function de(e) {
  if (typeof e != "object" || e === null || pe in e)
    return e;
  const t = ct(e);
  if (t !== Ht && t !== Gt)
    return e;
  var n = /* @__PURE__ */ new Map(), r = Ut(e), i = /* @__PURE__ */ $(0), s = ee, u = (f) => {
    if (ee === s)
      return f();
    var l = v, o = ee;
    O(null), lt(s);
    var _ = f();
    return O(l), lt(o), _;
  };
  return r && n.set("length", /* @__PURE__ */ $(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(f, l, o) {
        (!("value" in o) || o.configurable === !1 || o.enumerable === !1 || o.writable === !1) && rn();
        var _ = n.get(l);
        return _ === void 0 ? u(() => {
          var a = /* @__PURE__ */ $(o.value);
          return n.set(l, a), a;
        }) : W(_, o.value, !0), !0;
      },
      deleteProperty(f, l) {
        var o = n.get(l);
        if (o === void 0) {
          if (l in f) {
            const _ = u(() => /* @__PURE__ */ $(m));
            n.set(l, _), ge(i);
          }
        } else
          W(o, m), ge(i);
        return !0;
      },
      get(f, l, o) {
        if (l === pe)
          return e;
        var _ = n.get(l), a = l in f;
        if (_ === void 0 && (!a || ve(f, l)?.writable) && (_ = u(() => {
          var h = de(a ? f[l] : m), d = /* @__PURE__ */ $(h);
          return d;
        }), n.set(l, _)), _ !== void 0) {
          var c = J(_);
          return c === m ? void 0 : c;
        }
        return Reflect.get(f, l, o);
      },
      getOwnPropertyDescriptor(f, l) {
        var o = Reflect.getOwnPropertyDescriptor(f, l);
        if (o && "value" in o) {
          var _ = n.get(l);
          _ && (o.value = J(_));
        } else if (o === void 0) {
          var a = n.get(l), c = a?.v;
          if (a !== void 0 && c !== m)
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
        var o = n.get(l), _ = o !== void 0 && o.v !== m || Reflect.has(f, l);
        if (o !== void 0 || p !== null && (!_ || ve(f, l)?.writable)) {
          o === void 0 && (o = u(() => {
            var c = _ ? de(f[l]) : m, h = /* @__PURE__ */ $(c);
            return h;
          }), n.set(l, o));
          var a = J(o);
          if (a === m)
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
            d !== void 0 ? W(d, m) : h in f && (d = u(() => /* @__PURE__ */ $(m)), n.set(h + "", d));
          }
        if (a === void 0)
          (!c || ve(f, l)?.writable) && (a = u(() => /* @__PURE__ */ $(void 0)), W(a, de(o)), n.set(l, a));
        else {
          c = a.v !== m;
          var y = u(() => de(o));
          W(a, y);
        }
        var V = Reflect.getOwnPropertyDescriptor(f, l);
        if (V?.set && V.set.call(_, o), !c) {
          if (r && typeof l == "string") {
            var q = (
              /** @type {Source<number>} */
              n.get("length")
            ), ie = Number(l);
            Number.isInteger(ie) && ie >= q.v && W(q, ie + 1);
          }
          ge(i);
        }
        return !0;
      },
      ownKeys(f) {
        J(i);
        var l = Reflect.ownKeys(f).filter((a) => {
          var c = n.get(a);
          return c === void 0 || c.v !== m;
        });
        for (var [o, _] of n)
          _.v !== m && !(o in f) && l.push(o);
        return l;
      },
      setPrototypeOf() {
        sn();
      }
    }
  );
}
var it, On, At, Rt;
function Pn() {
  if (it === void 0) {
    it = window, On = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    At = ve(t, "firstChild").get, Rt = ve(t, "nextSibling").get, et(e) && (e[Qt] = void 0, e[Jt] = null, e[en] = void 0, e.__e = void 0), et(n) && (n[Ye] = void 0);
  }
}
function Ot(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function Cn(e) {
  return (
    /** @type {TemplateNode | null} */
    At.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function Pt(e) {
  return (
    /** @type {TemplateNode | null} */
    Rt.call(e)
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
function Dn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function G(e, t) {
  var n = p;
  n !== null && (n.f & j) !== 0 && (e |= j);
  var r = {
    ctx: T,
    deps: null,
    nodes: null,
    f: e | b | R,
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
    fe !== null ? fe.push(r) : ne.ensure().schedule(r);
  else if (t !== null) {
    try {
      ae(r);
    } catch (u) {
      throw I(r), u;
    }
    i.deps === null && i.teardown === null && i.nodes === null && i.first === i.last && // either `null`, or a singular child
    (i.f & he) === 0 && (i = i.first, (e & P) !== 0 && (e & Oe) !== 0 && i !== null && (i.f |= Oe));
  }
  if (i !== null && (i.parent = n, n !== null && Dn(i, n), v !== null && (v.f & S) !== 0 && (e & Y) === 0)) {
    var s = (
      /** @type {Derived} */
      v
    );
    (s.effects ??= []).push(i);
  }
  return r;
}
function Je() {
  return v !== null && !D;
}
function Nn(e) {
  const t = G(je, null);
  return w(t, E), t.teardown = e, t;
}
function Fn(e) {
  return G(ye | Zt, e);
}
function jn(e) {
  ne.ensure();
  const t = G(Y | he, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? Te(t, () => {
      I(t), r(void 0);
    }) : (I(t), r(void 0));
  });
}
function In(e) {
  return G(oe | he, e);
}
function Mn(e, t = 0) {
  return G(je | t, e);
}
function sr(e, t = [], n = [], r = []) {
  mn(r, t, n, (i) => {
    G(je, () => {
      e(...i.map(J));
    });
  });
}
function Ln(e, t = 0) {
  var n = G(P | t, e);
  return n;
}
function K(e) {
  return G(U | he, e);
}
function Ct(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = re, r = v;
    st(!0), O(null);
    try {
      t.call(null);
    } finally {
      st(n), O(r);
    }
  }
}
function Qe(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const i = n.ac;
    i !== null && Ie(() => {
      i.abort(me);
    });
    var r = n.next;
    (n.f & Y) !== 0 ? n.parent = null : I(n, t), n = r;
  }
}
function qn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & U) === 0 && I(t), t = n;
  }
}
function I(e, t = !0) {
  var n = !1;
  (t || (e.f & Xt) !== 0) && e.nodes !== null && e.nodes.end !== null && (Bn(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= tt, Qe(e, t && !n), we(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const s of r)
      s.stop();
  Ct(e), e.f ^= tt, e.f |= F;
  var i = e.parent;
  i !== null && i.first !== null && Dt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function Bn(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ Pt(e);
    e.remove(), e = n;
  }
}
function Dt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function Te(e, t, n = !0) {
  var r = [];
  Nt(e, r, !0);
  var i = () => {
    n && I(e), t && t();
  }, s = r.length;
  if (s > 0) {
    var u = () => --s || i();
    for (var f of r)
      f.out(u);
  } else
    i();
}
function Nt(e, t, n) {
  if ((e.f & j) === 0) {
    e.f ^= j;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const f of r)
        (f.is_global || n) && t.push(f);
    for (var i = e.first; i !== null; ) {
      var s = i.next;
      if ((i.f & Y) === 0) {
        var u = (i.f & Oe) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (i.f & U) !== 0 && (e.f & P) !== 0;
        Nt(i, t, u ? n : !1);
      }
      i = s;
    }
  }
}
function Un(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var i = n === r ? null : /* @__PURE__ */ Pt(n);
      t.append(n), n = i;
    }
}
let Ae = !1, re = !1;
function st(e) {
  re = e;
}
let v = null, D = !1;
function O(e) {
  v = e;
}
let p = null;
function L(e) {
  p = e;
}
let M = null;
function Yn(e) {
  v !== null && (M ??= /* @__PURE__ */ new Set()).add(e);
}
let k = null, x = 0, A = null;
function Vn(e) {
  A = e;
}
let Ft = 1, Z = 0, ee = Z;
function lt(e) {
  ee = e;
}
function jt() {
  return ++Ft;
}
function Ee(e) {
  var t = e.f;
  if ((t & b) !== 0)
    return !0;
  if (t & S && (e.f &= ~te), (t & N) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, i = 0; i < r; i++) {
      var s = n[i];
      if (Ee(
        /** @type {Derived} */
        s
      ) && mt(
        /** @type {Derived} */
        s
      ), s.wv > e.wv)
        return !0;
    }
    (t & R) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    C === null && w(e, E);
  }
  return !1;
}
function It(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(M !== null && M.has(e)))
    for (var i = 0; i < r.length; i++) {
      var s = r[i];
      (s.f & S) !== 0 ? It(
        /** @type {Derived} */
        s,
        t,
        !1
      ) : t === s && (n ? w(s, b) : (s.f & E) !== 0 && w(s, N), Ze(
        /** @type {Effect} */
        s
      ));
    }
}
function Mt(e) {
  var t = k, n = x, r = A, i = v, s = M, u = T, f = D, l = ee, o = e.f;
  k = /** @type {null | Value[]} */
  null, x = 0, A = null, v = (o & (U | Y)) === 0 ? e : null, M = null, ue(e.ctx), D = !1, ee = ++Z, e.ac !== null && (Ie(() => {
    e.ac.abort(me);
  }), e.ac = null);
  try {
    e.f |= Ce;
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
      if (Je() && (e.f & R) !== 0)
        for (d = x; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && x < c.length && (we(e, x), c.length = x);
    if (be() && A !== null && !D && c !== null && (e.f & (S | N | b)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      A.length; d++)
        It(
          A[d],
          /** @type {Effect} */
          e
        );
    if (i !== null && i !== e) {
      if (Z++, i.deps !== null)
        for (let y = 0; y < n; y += 1)
          i.deps[y].rv = Z;
      if (t !== null)
        for (const y of t)
          y.rv = Z;
      A !== null && (r === null ? r = A : r.push(.../** @type {Source[]} */
      A));
    }
    return (e.f & H) !== 0 && (e.f ^= H), a;
  } catch (y) {
    return vt(y);
  } finally {
    e.f ^= Ce, k = t, x = n, A = r, v = i, M = s, ue(u), D = f, ee = l;
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
  (k === null || !Re.call(k, t))) {
    var s = (
      /** @type {Derived} */
      t
    );
    (s.f & R) !== 0 && (s.f ^= R, s.f &= ~te), s.v !== m && Ke(s), s.ac !== null && Ie(() => {
      s.ac.abort(me), s.ac = null, w(s, b);
    }), xn(s), we(s, 0);
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
  if ((t & F) === 0) {
    w(e, E);
    var n = p, r = Ae;
    p = e, Ae = (t & (U | Y)) === 0;
    try {
      (t & (P | _t)) !== 0 ? qn(e) : Qe(e), Ct(e);
      var i = Mt(e);
      e.teardown = typeof i == "function" ? i : null, e.wv = Ft;
      var s;
      at && Bt && (e.f & b) !== 0 && e.deps;
    } finally {
      Ae = r, p = n;
    }
  }
}
function J(e) {
  var t = e.f, n = (t & S) !== 0;
  if (v !== null && !D) {
    var r = p !== null && (p.f & F) !== 0;
    if (!r && (M === null || !M.has(e))) {
      var i = v.deps;
      if ((v.f & Ce) !== 0)
        e.rv < Z && (e.rv = Z, k === null && i !== null && i[x] === e ? x++ : k === null ? k = [e] : k.push(e));
      else {
        v.deps ??= [], Re.call(v.deps, e) || v.deps.push(e);
        var s = e.reactions;
        s === null ? e.reactions = [v] : Re.call(s, v) || s.push(v);
      }
    }
  }
  if (re && Q.has(e))
    return Q.get(e);
  if (n) {
    var u = (
      /** @type {Derived} */
      e
    );
    if (re) {
      var f = u.v;
      return ((u.f & E) === 0 && u.reactions !== null || qt(u)) && (f = We(u)), Q.set(u, f), f;
    }
    var l = (u.f & R) === 0 && !D && v !== null && (Ae || (v.f & R) !== 0), o = (u.f & ce) === 0;
    Ee(u) && (l && (u.f |= R), mt(u)), l && !o && (bt(u), Lt(u));
  }
  if (C?.has(e))
    return C.get(e);
  if ((e.f & H) !== 0)
    throw e.v;
  return e.v;
}
function Lt(e) {
  if (e.f |= R, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & S) !== 0 && (t.f & R) === 0 && (bt(
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
    if (Q.has(t) || (t.f & S) !== 0 && qt(
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
      ze(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && pe in n && ze(n);
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
    const n = ct(e);
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
const Se = /* @__PURE__ */ Symbol("events"), Hn = /* @__PURE__ */ new Set(), ft = /* @__PURE__ */ new Set();
let ot = null;
function ut(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, i = e.composedPath?.() || [], s = (
    /** @type {null | Element} */
    i[0] || e.target
  );
  ot = e;
  var u = 0, f = ot === e && e[Se];
  if (f) {
    var l = i.indexOf(f);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[Se] = t;
      return;
    }
    var o = i.indexOf(t);
    if (o === -1)
      return;
    l <= o && (u = l);
  }
  if (s = /** @type {Element} */
  i[u] || e.target, s !== t) {
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
          var d = s[Se]?.[r];
          d != null && (!/** @type {any} */
          s.disabled || // DOM could've been updated already by the time this is reached, so we check this as well
          // -> the target could not have been disabled because it emits the event in the first place
          e.target === s) && d.call(s, e);
        } catch (y) {
          c ? h.push(y) : c = y;
        }
        if (e.cancelBubble) break;
        u++, s = u < i.length ? (
          /** @type {Element} */
          i[u]
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
      e[Se] = t, delete e.currentTarget, O(_), L(a);
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
  (e[Ye] ??= e.nodeValue) && (e[Ye] = n, e.nodeValue = `${n}`);
}
function or(e, t) {
  return Wn(e, t);
}
const ke = /* @__PURE__ */ new Map();
function Wn(e, { target: t, anchor: n, props: r = {}, events: i, context: s, intro: u = !0, transformError: f }) {
  Pn();
  var l = void 0, o = jn(() => {
    var _ = n ?? t.appendChild(Ot());
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
      f
    );
    var a = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var y = h[d];
        if (!a.has(y)) {
          a.add(y);
          var V = Kn(y);
          for (const Le of [t, document]) {
            var q = ke.get(Le);
            q === void 0 && (q = /* @__PURE__ */ new Map(), ke.set(Le, q));
            var ie = q.get(y);
            ie === void 0 ? (Le.addEventListener(y, ut, { passive: V }), q.set(y, 1)) : q.set(y, ie + 1);
          }
        }
      }
    };
    return c(Vt(Hn)), ft.add(c), () => {
      for (var h of a)
        for (const V of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            ke.get(V)
          ), y = (
            /** @type {number} */
            d.get(h)
          );
          --y == 0 ? (V.removeEventListener(h, ut), d.delete(h), d.size === 0 && ke.delete(V)) : d.set(h, y);
        }
      ft.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return He.set(l, o), l;
}
let He = /* @__PURE__ */ new WeakMap();
function ur(e, t) {
  const n = He.get(e);
  return n ? (He.delete(e), n(t)) : Promise.resolve();
}
export {
  F as D,
  tr as L,
  Jn as P,
  pe as S,
  er as T,
  p as a,
  ve as b,
  ir as c,
  J as d,
  Xn as e,
  Zn as f,
  Cn as g,
  Qn as h,
  On as i,
  Sn as j,
  re as k,
  Ge as l,
  lr as m,
  fr as n,
  rr as o,
  de as p,
  or as q,
  ur as r,
  W as s,
  sr as t,
  zn as u
};
