const lr = 1, fr = 2, ar = 4, ur = 8, or = 16, cr = 1, hr = 2, _r = 4, dr = 8, vr = 16, pr = 1, gr = 2, b = /* @__PURE__ */ Symbol("uninitialized"), Jt = "http://www.w3.org/1999/xhtml", yr = "http://www.w3.org/2000/svg", wr = "@attach", pt = !1;
var Qt = Array.isArray, en = Array.prototype.indexOf, Oe = Array.prototype.includes, tn = Array.from, nn = Object.defineProperty, we = Object.getOwnPropertyDescriptor, rn = Object.getOwnPropertyDescriptors, sn = Object.prototype, ln = Array.prototype, gt = Object.getPrototypeOf, ft = Object.isExtensible;
const fn = () => {
};
function Er(e) {
  return e();
}
function an(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function yt() {
  var e, t, n = new Promise((r, s) => {
    e = r, t = s;
  });
  return { promise: n, resolve: e, reject: t };
}
const T = 2, ce = 4, Se = 8, Xe = 1 << 24, P = 16, H = 32, W = 64, qe = 128, C = 512, m = 1024, S = 2048, F = 4096, R = 8192, D = 16384, _e = 32768, at = 1 << 25, me = 65536, Ie = 1 << 17, un = 1 << 18, de = 1 << 19, wt = 1 << 20, mr = 1 << 25, se = 65536, Pe = 1 << 21, oe = 1 << 22, K = 1 << 23, ee = /* @__PURE__ */ Symbol("$state"), br = /* @__PURE__ */ Symbol("legacy props"), Sr = /* @__PURE__ */ Symbol(""), on = /* @__PURE__ */ Symbol("attributes"), cn = /* @__PURE__ */ Symbol("class"), hn = /* @__PURE__ */ Symbol("style"), Ue = /* @__PURE__ */ Symbol("text"), Fe = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}(), Ar = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  !!globalThis.document?.contentType && /* @__PURE__ */ globalThis.document.contentType.includes("xml")
);
function _n() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function kr(e, t, n) {
  throw new Error("https://svelte.dev/e/each_key_duplicate");
}
function dn(e) {
  throw new Error("https://svelte.dev/e/effect_in_teardown");
}
function vn() {
  throw new Error("https://svelte.dev/e/effect_in_unowned_derived");
}
function pn(e) {
  throw new Error("https://svelte.dev/e/effect_orphan");
}
function gn() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function xr(e) {
  throw new Error("https://svelte.dev/e/props_invalid_value");
}
function yn() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function wn() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function En() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function mn() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function bn() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function Cr() {
  console.warn("https://svelte.dev/e/select_multiple_invalid_value");
}
function Sn() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function Et(e) {
  return e === this.v;
}
function Tn(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function mt(e) {
  return !Tn(e, this.v);
}
let Le = !1, An = !1;
function Rr() {
  Le = !0;
}
let E = null;
function he(e) {
  E = e;
}
function kn(e, t = !1, n) {
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
    l: Le && !t ? { s: null, u: null, $: [] } : null
  };
}
function xn(e) {
  var t = (
    /** @type {ComponentContext} */
    E
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      Bt(r);
  }
  return t.i = !0, E = t.p, /** @type {T} */
  {};
}
function Te() {
  return !Le || E !== null && E.l === null;
}
let ae = [];
function Cn() {
  var e = ae;
  ae = [], an(e);
}
function te(e) {
  if (ae.length === 0) {
    var t = ae;
    queueMicrotask(() => {
      t === ae && Cn();
    });
  }
  ae.push(e);
}
function bt(e) {
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
      if ((t.f & qe) !== 0) {
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
const Rn = -7169;
function w(e, t) {
  e.f = e.f & Rn | t;
}
function Ze(e) {
  (e.f & C) !== 0 || e.deps === null ? w(e, m) : w(e, F);
}
function St(e) {
  if (e !== null)
    for (const t of e)
      (t.f & T) === 0 || (t.f & se) === 0 || (t.f ^= se, St(
        /** @type {Derived} */
        t.deps
      ));
}
function Tt(e, t, n) {
  (e.f & S) !== 0 ? t.add(e) : (e.f & F) !== 0 && n.add(e), St(e.deps), w(e, m);
}
function On(e) {
  let t = 0, n = Ae(0), r;
  return () => {
    rt() && (z(n), st(() => (t === 0 && (r = lt(() => e(() => Ee(n)))), t += 1, () => {
      te(() => {
        t -= 1, t === 0 && (r?.(), r = void 0, Ee(n));
      });
    })));
  };
}
var In = me | de;
function Pn(e, t, n, r) {
  new Nn(e, t, n, r);
}
class Nn {
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
  #y = On(() => (this.#u = Ae(this.#_), () => {
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
      u.b = this, u.f |= qe, r(i);
    }, this.parent = /** @type {Effect} */
    p.b, this.transform_error = s ?? this.parent?.transform_error ?? ((i) => i), this.#n = Kn(() => {
      this.#w();
    }, In);
  }
  #g() {
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
    t && (this.is_pending = !0, this.#e = J(() => t(this.#i)), te(() => {
      var n = this.#t = document.createDocumentFragment(), r = Dt();
      n.append(r), this.#l = this.#E(() => J(() => this.#h(r))), this.#f === 0 && (this.#i.before(n), this.#t = null, Ce(
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
      if (this.is_pending = this.has_pending_snippet(), this.#f = 0, this.#_ = 0, this.#l = J(() => {
        this.#h(this.#i);
      }), this.#f > 0) {
        var t = this.#t = document.createDocumentFragment();
        Zn(this.#l, t);
        const n = (
          /** @type {(anchor: Node) => void} */
          this.#r.pending
        );
        this.#e = J(() => n(this.#i));
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
    this.is_pending = !1, t.transfer_effects(this.#o, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(t) {
    Tt(t, this.#o, this.#p);
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
    I(this.#n), O(this.#n), he(this.#n.ctx);
    try {
      return X.ensure(), t();
    } catch (i) {
      return bt(i), null;
    } finally {
      I(n), O(r), he(s);
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
    this.#f += t, this.#f === 0 && (this.#c(n), this.#e && Ce(this.#e, () => {
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
    this.#d(t, n), this.#_ += t, !(!this.#u || this.#a) && (this.#a = !0, te(() => {
      this.#a = !1, this.#u && De(this.#u, this.#_);
    }));
  }
  get_effect_pending() {
    return this.#y(), z(
      /** @type {Source<number>} */
      this.#u
    );
  }
  /** @param {unknown} error */
  error(t) {
    if (!this.#r.onerror && !this.#r.failed)
      throw t;
    g?.is_fork ? (this.#l && g.skip_effect(this.#l), this.#e && g.skip_effect(this.#e), this.#s && g.skip_effect(this.#s), g.oncommit(() => {
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
        Sn();
        return;
      }
      s = !0, i && mn(), this.#s !== null && Ce(this.#s, () => {
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
            f.b = this, f.f |= qe, r(
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
    te(() => {
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
function Mn(e, t, n, r) {
  const s = Te() ? Je : Ln;
  var i = e.filter((h) => !h.settled), u = t.map(s);
  if (n.length === 0 && i.length === 0) {
    r(u);
    return;
  }
  var a = (
    /** @type {Effect} */
    p
  ), l = Dn(), f = i.length === 1 ? i[0].promise : i.length > 1 ? Promise.all(i.map((h) => h.promise)) : null;
  function _(h) {
    if ((a.f & D) === 0) {
      l();
      try {
        r([...u, ...h]);
      } catch (d) {
        $(d, a);
      }
      Ne();
    }
  }
  var o = At();
  if (n.length === 0) {
    f.then(() => _([])).finally(o);
    return;
  }
  function c() {
    Promise.all(n.map((h) => /* @__PURE__ */ Fn(h))).then(_).catch((h) => $(h, a)).finally(o);
  }
  f ? f.then(() => {
    l(), c(), Ne();
  }) : c();
}
function Dn() {
  var e = (
    /** @type {Effect} */
    p
  ), t = v, n = E, r = (
    /** @type {Batch} */
    g
  );
  return function(i = !0) {
    I(e), O(t), he(n), i && (e.f & D) === 0 && (r?.activate(), r?.apply());
  };
}
function Ne(e = !0) {
  I(null), O(null), he(null), e && g?.deactivate();
}
function At() {
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
function Je(e) {
  var t = T | S;
  return p !== null && (p.f |= de), {
    ctx: E,
    deps: null,
    effects: null,
    equals: Et,
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
function Fn(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && _n();
  var s = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), i = Ae(
    /** @type {V} */
    b
  ), u = !v, a = /* @__PURE__ */ new Set();
  return zn(() => {
    var l = (
      /** @type {Effect} */
      p
    ), f = yt();
    s = f.promise;
    try {
      Promise.resolve(e()).then(f.resolve, (h) => {
        h !== Fe && f.reject(h);
      }).finally(Ne);
    } catch (h) {
      f.reject(h), Ne();
    }
    var _ = (
      /** @type {Batch} */
      g
    );
    if (u) {
      if ((l.f & _e) !== 0)
        var o = At();
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
      o?.(), a.delete(f), d !== pe && (_.activate(), d ? (i.f |= K, De(i, d)) : ((i.f & K) !== 0 && (i.f ^= K), De(i, h)), _.deactivate());
    };
    f.promise.then(c, (h) => c(null, h || "unknown"));
  }), jt(() => {
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
function Or(e) {
  const t = /* @__PURE__ */ Je(e);
  return Yt(t), t;
}
// @__NO_SIDE_EFFECTS__
function Ln(e) {
  const t = /* @__PURE__ */ Je(e);
  return t.equals = mt, t;
}
function jn(e) {
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
function Qe(e) {
  var t, n = p, r = e.parent;
  if (!Z && r !== null && e.v !== b && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (D | R)) !== 0)
    return bn(), e.v;
  I(r);
  try {
    e.f &= ~se, jn(e), t = Kt(e);
  } finally {
    I(n);
  }
  return t;
}
function kt(e) {
  var t = Qe(e);
  if (!e.equals(t) && (e.wv = $t(), (!g?.is_fork || e.deps === null) && (g !== null ? (g.capture(e, t, !0), Ye?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    w(e, m);
    return;
  }
  Z || (N !== null ? (rt() || g?.is_fork) && N.set(e, t) : Ze(e));
}
function Bn(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac?.abort(Fe), t.fn !== null && (t.teardown = fn), t.ac = null, be(t, 0), it(t));
}
function xt(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && ie(t);
}
let He = null, fe = null, g = null, Ye = null, N = null, Ge = null, Ve = !1, ue = null, xe = null;
var ut = 0;
let Hn = 1;
class X {
  id = Hn++;
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
    fe === null ? He = fe = this : (fe.#r = this, this.#v = fe), fe = this;
  }
  #y() {
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
  #g() {
    this.#i = !0, ut++ > 1e3 && (this.#d(), Vn());
    for (const l of this.#f)
      this.#a.delete(l), w(l, S), this.schedule(l);
    for (const l of this.#a)
      w(l, F), this.schedule(l);
    const t = this.#t;
    this.#t = [], this.apply();
    var n = ue = [], r = [], s = xe = [];
    for (const l of t)
      try {
        this.#m(l, n, r);
      } catch (f) {
        throw Ot(l), this.#y() || this.discard(), f;
      }
    if (g = null, s.length > 0) {
      var i = X.ensure();
      for (const l of s)
        i.schedule(l);
    }
    if (ue = null, xe = null, this.#y()) {
      this.#c(r), this.#c(n);
      for (const [l, f] of this.#o)
        Rt(l, f);
      s.length > 0 && /** @type {unknown} */
      g.#g();
      return;
    }
    const u = this.#b();
    if (u) {
      this.#c(r), this.#c(n), u.#w(this);
      return;
    }
    this.#f.clear(), this.#a.clear();
    for (const l of this.#h) l(this);
    this.#h.clear(), Ye = this, ot(r), ot(n), Ye = null, this.#s?.resolve();
    var a = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      g
    );
    if (this.#l === 0 && (this.#t.length === 0 || a !== null) && this.#d(), this.#t.length > 0)
      if (a !== null) {
        const l = a;
        l.#t.push(...this.#t.filter((f) => !l.#t.includes(f)));
      } else
        a = this;
    a !== null && a.#g();
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
        u ? s.f ^= m : (i & ce) !== 0 ? n.push(s) : ve(s) && ((i & P) !== 0 && this.#a.add(s), ie(s));
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
            i & (oe | P) && !this.async_deriveds.has(u) && (this.#a.delete(u), w(u, S), this.schedule(u));
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
      Tt(t[n], this.#f, this.#a);
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
    g = this;
  }
  deactivate() {
    g = null, N = null;
  }
  flush() {
    try {
      Ve = !0, g = this, this.#g();
    } finally {
      ut = 0, Ge = null, ue = null, xe = null, Ve = !1, g = null, N = null, ne.clear();
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
                (h.f & (P | oe)) !== 0 ? o.schedule(h) : o.#c([h]);
              });
          o.activate();
          var u = /* @__PURE__ */ new Set(), a = /* @__PURE__ */ new Map();
          for (var l of n)
            Ct(l, i, u, a);
          a = /* @__PURE__ */ new Map();
          var f = [...o.current].filter(([c, h]) => {
            const d = this.current.get(c);
            return d ? d[0] !== h[0] || d[1] !== h[1] : !0;
          }).map(([c]) => c);
          if (f.length > 0)
            for (const c of this.#_)
              (c.f & (D | R | Ie)) === 0 && et(c, f, a) && ((c.f & (oe | P)) !== 0 ? (w(c, S), o.schedule(c)) : o.#f.add(c));
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
    this.#u || (this.#u = !0, te(() => {
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
    return (this.#s ??= yt()).promise;
  }
  static ensure() {
    if (g === null) {
      const t = g = new X();
      Ve || te(() => {
        t.#i || t.flush();
      });
    }
    return g;
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
    if (Ge = t, t.b?.is_pending && (t.f & (ce | Se | Xe)) !== 0 && (t.f & _e) === 0) {
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
      t === null ? He = n : t.#r = n, n === null ? fe = t : n.#v = t, this.linked = !1;
    }
  }
}
function Vn() {
  try {
    gn();
  } catch (e) {
    $(e, Ge);
  }
}
let q = null;
function ot(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (D | R)) === 0 && ve(r) && (q = /* @__PURE__ */ new Set(), ie(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && Vt(r), q?.size > 0)) {
        ne.clear();
        for (const s of q) {
          if ((s.f & (D | R)) !== 0) continue;
          const i = [s];
          let u = s.parent;
          for (; u !== null; )
            q.has(u) && (q.delete(u), i.push(u)), u = u.parent;
          for (let a = i.length - 1; a >= 0; a--) {
            const l = i[a];
            (l.f & (D | R)) === 0 && ie(l);
          }
        }
        q.clear();
      }
    }
    q = null;
  }
}
function Ct(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const s of e.reactions) {
      const i = s.f;
      (i & T) !== 0 ? Ct(
        /** @type {Derived} */
        s,
        t,
        n,
        r
      ) : (i & (oe | P)) !== 0 && (i & S) === 0 && et(s, t, r) && (w(s, S), tt(
        /** @type {Effect} */
        s
      ));
    }
}
function et(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const s of e.deps) {
      if (Oe.call(t, s))
        return !0;
      if ((s.f & T) !== 0 && et(
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
function tt(e) {
  g.schedule(e);
}
function Rt(e, t) {
  if (!((e.f & H) !== 0 && (e.f & m) !== 0)) {
    (e.f & S) !== 0 ? t.d.push(e) : (e.f & F) !== 0 && t.m.push(e), w(e, m);
    for (var n = e.first; n !== null; )
      Rt(n, t), n = n.next;
  }
}
function Ot(e) {
  w(e, m);
  for (var t = e.first; t !== null; )
    Ot(t), t = t.next;
}
let Me = /* @__PURE__ */ new Set();
const ne = /* @__PURE__ */ new Map();
let It = !1;
function Ae(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: Et,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function Y(e, t) {
  const n = Ae(e);
  return Yt(n), n;
}
// @__NO_SIDE_EFFECTS__
function Ir(e, t = !1, n = !0) {
  const r = Ae(e);
  return t || (r.equals = mt), Le && n && E !== null && E.l !== null && (E.l.s ??= []).push(r), r;
}
function Pr(e, t) {
  return G(
    e,
    lt(() => z(e))
  ), t;
}
function G(e, t, n = !1) {
  v !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!M || (v.f & Ie) !== 0) && Te() && (v.f & (T | P | oe | Ie)) !== 0 && (B === null || !B.has(e)) && En();
  let r = n ? ge(t) : t;
  return De(e, r, xe);
}
function De(e, t, n = null) {
  if (!e.equals(t)) {
    ne.set(e, Z ? t : e.v);
    var r = X.ensure();
    if (r.capture(e, t), (e.f & T) !== 0) {
      const s = (
        /** @type {Derived} */
        e
      );
      (e.f & S) !== 0 && Qe(s), N === null && Ze(s);
    }
    e.wv = $t(), Pt(e, S, n), Te() && p !== null && (p.f & m) !== 0 && (p.f & (H | W)) === 0 && (x === null ? Jn([e]) : x.push(e)), !r.is_fork && Me.size > 0 && !It && qn();
  }
  return t;
}
function qn() {
  It = !1;
  for (const e of Me) {
    (e.f & m) !== 0 && w(e, F);
    let t;
    try {
      t = ve(e);
    } catch {
      t = !0;
    }
    t && ie(e);
  }
  Me.clear();
}
function Ee(e) {
  G(e, e.v + 1);
}
function Pt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var s = Te(), i = r.length, u = 0; u < i; u++) {
      var a = r[u], l = a.f;
      if (!(!s && a === p)) {
        var f = (l & S) === 0;
        if (f && w(a, t), (l & Ie) !== 0)
          Me.add(
            /** @type {Effect} */
            a
          );
        else if ((l & T) !== 0) {
          var _ = (
            /** @type {Derived} */
            a
          );
          N?.delete(_), (l & se) === 0 && (l & C && (p === null || (p.f & Pe) === 0) && (a.f |= se), Pt(_, F, n));
        } else if (f) {
          var o = (
            /** @type {Effect} */
            a
          );
          (l & P) !== 0 && q !== null && q.add(o), n !== null ? n.push(o) : tt(o);
        }
      }
    }
}
function ge(e) {
  if (typeof e != "object" || e === null || ee in e)
    return e;
  const t = gt(e);
  if (t !== sn && t !== ln)
    return e;
  var n = /* @__PURE__ */ new Map(), r = Qt(e), s = /* @__PURE__ */ Y(0), i = re, u = (a) => {
    if (re === i)
      return a();
    var l = v, f = re;
    O(null), dt(i);
    var _ = a();
    return O(l), dt(f), _;
  };
  return r && n.set("length", /* @__PURE__ */ Y(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(a, l, f) {
        (!("value" in f) || f.configurable === !1 || f.enumerable === !1 || f.writable === !1) && yn();
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
            n.set(l, _), Ee(s);
          }
        } else
          G(f, b), Ee(s);
        return !0;
      },
      get(a, l, f) {
        if (l === ee)
          return e;
        var _ = n.get(l), o = l in a;
        if (_ === void 0 && (!o || we(a, l)?.writable) && (_ = u(() => {
          var h = ge(o ? a[l] : b), d = /* @__PURE__ */ Y(h);
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
        if (l === ee)
          return !0;
        var f = n.get(l), _ = f !== void 0 && f.v !== b || Reflect.has(a, l);
        if (f !== void 0 || p !== null && (!_ || we(a, l)?.writable)) {
          f === void 0 && (f = u(() => {
            var c = _ ? ge(a[l]) : b, h = /* @__PURE__ */ Y(c);
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
          (!c || we(a, l)?.writable) && (o = u(() => /* @__PURE__ */ Y(void 0)), G(o, ge(f)), n.set(l, o));
        else {
          c = o.v !== b;
          var y = u(() => ge(f));
          G(o, y);
        }
        var U = Reflect.getOwnPropertyDescriptor(a, l);
        if (U?.set && U.set.call(_, f), !c) {
          if (r && typeof l == "string") {
            var V = (
              /** @type {Source<number>} */
              n.get("length")
            ), le = Number(l);
            Number.isInteger(le) && le >= V.v && G(V, le + 1);
          }
          Ee(s);
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
        wn();
      }
    }
  );
}
function ct(e) {
  try {
    if (e !== null && typeof e == "object" && ee in e)
      return e[ee];
  } catch {
  }
  return e;
}
function Nr(e, t) {
  return Object.is(ct(e), ct(t));
}
var ht, Un, Nt, Mt;
function Yn() {
  if (ht === void 0) {
    ht = window, Un = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    Nt = we(t, "firstChild").get, Mt = we(t, "nextSibling").get, ft(e) && (e[cn] = void 0, e[on] = null, e[hn] = void 0, e.__e = void 0), ft(n) && (n[Ue] = void 0);
  }
}
function Dt(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
  return (
    /** @type {TemplateNode | null} */
    Nt.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function je(e) {
  return (
    /** @type {TemplateNode | null} */
    Mt.call(e)
  );
}
function Mr(e, t) {
  return /* @__PURE__ */ Ft(e);
}
function Dr(e, t = !1) {
  {
    var n = /* @__PURE__ */ Ft(e);
    return n instanceof Comment && n.data === "" ? /* @__PURE__ */ je(n) : n;
  }
}
function Fr(e, t = 1, n = !1) {
  let r = e;
  for (; t--; )
    r = /** @type {TemplateNode} */
    /* @__PURE__ */ je(r);
  return r;
}
function Lr(e) {
  e.textContent = "";
}
function jr() {
  return !1;
}
function Br(e, t, n) {
  return t == null || t === Jt ? (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  ) : (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElementNS(t, e, { is: n }) : document.createElementNS(t, e)
  );
}
function nt(e) {
  var t = v, n = p;
  O(null), I(null);
  try {
    return e();
  } finally {
    O(t), I(n);
  }
}
function Lt(e) {
  p === null && (v === null && pn(), vn()), Z && dn();
}
function Gn(e, t) {
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
  g?.register_created_effect(r);
  var s = r;
  if ((e & ce) !== 0)
    ue !== null ? ue.push(r) : X.ensure().schedule(r);
  else if (t !== null) {
    try {
      ie(r);
    } catch (u) {
      throw j(r), u;
    }
    s.deps === null && s.teardown === null && s.nodes === null && s.first === s.last && // either `null`, or a singular child
    (s.f & de) === 0 && (s = s.first, (e & P) !== 0 && (e & me) !== 0 && s !== null && (s.f |= me));
  }
  if (s !== null && (s.parent = n, n !== null && Gn(s, n), v !== null && (v.f & T) !== 0 && (e & W) === 0)) {
    var i = (
      /** @type {Derived} */
      v
    );
    (i.effects ??= []).push(s);
  }
  return r;
}
function rt() {
  return v !== null && !M;
}
function jt(e) {
  const t = L(Se, null);
  return w(t, m), t.teardown = e, t;
}
function Hr(e) {
  Lt();
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
    return Bt(e);
}
function Bt(e) {
  return L(ce | wt, e);
}
function Vr(e) {
  return Lt(), L(Se | wt, e);
}
function $n(e) {
  X.ensure();
  const t = L(W | de, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? Ce(t, () => {
      j(t), r(void 0);
    }) : (j(t), r(void 0));
  });
}
function qr(e) {
  return L(ce, e);
}
function Ur(e, t) {
  var n = (
    /** @type {ComponentContextLegacy} */
    E
  ), r = { effect: null, ran: !1, deps: e };
  n.l.$.push(r), r.effect = st(() => {
    if (e(), !r.ran) {
      r.ran = !0;
      var s = (
        /** @type {Effect} */
        p
      );
      try {
        I(s.parent), lt(t);
      } finally {
        I(s);
      }
    }
  });
}
function Yr() {
  var e = (
    /** @type {ComponentContextLegacy} */
    E
  );
  st(() => {
    for (var t of e.l.$) {
      t.deps();
      var n = t.effect;
      (n.f & m) !== 0 && n.deps !== null && w(n, F), ve(n) && ie(n), t.ran = !1;
    }
  });
}
function zn(e) {
  return L(oe | de, e);
}
function st(e, t = 0) {
  return L(Se | t, e);
}
function Gr(e, t = [], n = [], r = []) {
  Mn(r, t, n, (s) => {
    L(Se, () => {
      e(...s.map(z));
    });
  });
}
function Kn(e, t = 0) {
  var n = L(P | t, e);
  return n;
}
function $r(e, t = 0) {
  var n = L(Xe | t, e);
  return n;
}
function J(e) {
  return L(H | de, e);
}
function Ht(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = Z, r = v;
    _t(!0), O(null);
    try {
      t.call(null);
    } finally {
      _t(n), O(r);
    }
  }
}
function it(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const s = n.ac;
    s !== null && nt(() => {
      s.abort(Fe);
    });
    var r = n.next;
    (n.f & W) !== 0 ? n.parent = null : j(n, t), n = r;
  }
}
function Wn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & H) === 0 && j(t), t = n;
  }
}
function j(e, t = !0) {
  var n = !1;
  (t || (e.f & un) !== 0) && e.nodes !== null && e.nodes.end !== null && (Xn(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= at, it(e, t && !n), be(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const i of r)
      i.stop();
  Ht(e), e.f ^= at, e.f |= D;
  var s = e.parent;
  s !== null && s.first !== null && Vt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function Xn(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ je(e);
    e.remove(), e = n;
  }
}
function Vt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function Ce(e, t, n = !0) {
  var r = [];
  qt(e, r, !0);
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
function qt(e, t, n) {
  if ((e.f & R) === 0) {
    e.f ^= R;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const a of r)
        (a.is_global || n) && t.push(a);
    for (var s = e.first; s !== null; ) {
      var i = s.next;
      if ((s.f & W) === 0) {
        var u = (s.f & me) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (s.f & H) !== 0 && (e.f & P) !== 0;
        qt(s, t, u ? n : !1);
      }
      s = i;
    }
  }
}
function zr(e) {
  Ut(e, !0);
}
function Ut(e, t) {
  if ((e.f & R) !== 0) {
    e.f ^= R, (e.f & m) === 0 && (w(e, S), X.ensure().schedule(e));
    for (var n = e.first; n !== null; ) {
      var r = n.next, s = (n.f & me) !== 0 || (n.f & H) !== 0;
      Ut(n, s ? t : !1), n = r;
    }
    var i = e.nodes && e.nodes.t;
    if (i !== null)
      for (const u of i)
        (u.is_global || t) && u.in();
  }
}
function Zn(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var s = n === r ? null : /* @__PURE__ */ je(n);
      t.append(n), n = s;
    }
}
let Re = !1, Z = !1;
function _t(e) {
  Z = e;
}
let v = null, M = !1;
function O(e) {
  v = e;
}
let p = null;
function I(e) {
  p = e;
}
let B = null;
function Yt(e) {
  v !== null && (B ??= /* @__PURE__ */ new Set()).add(e);
}
let A = null, k = 0, x = null;
function Jn(e) {
  x = e;
}
let Gt = 1, Q = 0, re = Q;
function dt(e) {
  re = e;
}
function $t() {
  return ++Gt;
}
function ve(e) {
  var t = e.f;
  if ((t & S) !== 0)
    return !0;
  if (t & T && (e.f &= ~se), (t & F) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, s = 0; s < r; s++) {
      var i = n[s];
      if (ve(
        /** @type {Derived} */
        i
      ) && kt(
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
function zt(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(B !== null && B.has(e)))
    for (var s = 0; s < r.length; s++) {
      var i = r[s];
      (i.f & T) !== 0 ? zt(
        /** @type {Derived} */
        i,
        t,
        !1
      ) : t === i && (n ? w(i, S) : (i.f & m) !== 0 && w(i, F), tt(
        /** @type {Effect} */
        i
      ));
    }
}
function Kt(e) {
  var t = A, n = k, r = x, s = v, i = B, u = E, a = M, l = re, f = e.f;
  A = /** @type {null | Value[]} */
  null, k = 0, x = null, v = (f & (H | W)) === 0 ? e : null, B = null, he(e.ctx), M = !1, re = ++Q, e.ac !== null && (nt(() => {
    e.ac.abort(Fe);
  }), e.ac = null);
  try {
    e.f |= Pe;
    var _ = (
      /** @type {Function} */
      e.fn
    ), o = _();
    e.f |= _e;
    var c = e.deps, h = g?.is_fork;
    if (A !== null) {
      var d;
      if (h || be(e, k), c !== null && k > 0)
        for (c.length = k + A.length, d = 0; d < A.length; d++)
          c[k + d] = A[d];
      else
        e.deps = c = A;
      if (rt() && (e.f & C) !== 0)
        for (d = k; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !h && c !== null && k < c.length && (be(e, k), c.length = k);
    if (Te() && x !== null && !M && c !== null && (e.f & (T | F | S)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      x.length; d++)
        zt(
          x[d],
          /** @type {Effect} */
          e
        );
    if (s !== null && s !== e) {
      if (Q++, s.deps !== null)
        for (let y = 0; y < n; y += 1)
          s.deps[y].rv = Q;
      if (t !== null)
        for (const y of t)
          y.rv = Q;
      x !== null && (r === null ? r = x : r.push(.../** @type {Source[]} */
      x));
    }
    return (e.f & K) !== 0 && (e.f ^= K), o;
  } catch (y) {
    return bt(y);
  } finally {
    e.f ^= Pe, A = t, k = n, x = r, v = s, B = i, he(u), M = a, re = l;
  }
}
function Qn(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = en.call(n, e);
    if (r !== -1) {
      var s = n.length - 1;
      s === 0 ? n = t.reactions = null : (n[r] = n[s], n.pop());
    }
  }
  if (n === null && (t.f & T) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (A === null || !Oe.call(A, t))) {
    var i = (
      /** @type {Derived} */
      t
    );
    (i.f & C) !== 0 && (i.f ^= C, i.f &= ~se), i.v !== b && Ze(i), Bn(i), be(i, 0);
  }
}
function be(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      Qn(e, n[r]);
}
function ie(e) {
  var t = e.f;
  if ((t & D) === 0) {
    w(e, m);
    var n = p, r = Re;
    p = e, Re = !0;
    try {
      (t & (P | Xe)) !== 0 ? Wn(e) : it(e), Ht(e);
      var s = Kt(e);
      e.teardown = typeof s == "function" ? s : null, e.wv = Gt;
      var i;
      pt && An && (e.f & S) !== 0 && e.deps;
    } finally {
      Re = r, p = n;
    }
  }
}
function z(e) {
  var t = e.f, n = (t & T) !== 0;
  if (v !== null && !M) {
    var r = p !== null && (p.f & D) !== 0;
    if (!r && (B === null || !B.has(e))) {
      var s = v.deps;
      if ((v.f & Pe) !== 0)
        e.rv < Q && (e.rv = Q, A === null && s !== null && s[k] === e ? k++ : A === null ? A = [e] : A.push(e));
      else {
        v.deps ??= [], Oe.call(v.deps, e) || v.deps.push(e);
        var i = e.reactions;
        i === null ? e.reactions = [v] : Oe.call(i, v) || i.push(v);
      }
    }
  }
  if (Z && ne.has(e))
    return ne.get(e);
  if (n) {
    var u = (
      /** @type {Derived} */
      e
    );
    if (Z) {
      var a = u.v;
      return ((u.f & m) === 0 && u.reactions !== null || Xt(u)) && (a = Qe(u)), ne.set(u, a), a;
    }
    var l = (u.f & C) === 0 && !M && v !== null && (Re || (v.f & C) !== 0), f = (u.f & _e) === 0;
    ve(u) && (l && (u.f |= C), kt(u)), l && !f && (xt(u), Wt(u));
  }
  if (N?.has(e))
    return N.get(e);
  if ((e.f & K) !== 0)
    throw e.v;
  return e.v;
}
function Wt(e) {
  if (e.f |= C, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & T) !== 0 && (t.f & C) === 0 && (xt(
        /** @type {Derived} */
        t
      ), Wt(
        /** @type {Derived} */
        t
      ));
}
function Xt(e) {
  if (e.v === b) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (ne.has(t) || (t.f & T) !== 0 && Xt(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function lt(e) {
  var t = M;
  try {
    return M = !0, e();
  } finally {
    M = t;
  }
}
function Kr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (ee in e)
      $e(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && ee in n && $e(n);
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
    const n = gt(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = rn(n);
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
const ye = /* @__PURE__ */ Symbol("events"), Zt = /* @__PURE__ */ new Set(), ze = /* @__PURE__ */ new Set();
function er(e, t, n, r = {}) {
  function s(i) {
    if (r.capture || Ke.call(t, i), !i.cancelBubble)
      return nt(() => n?.call(this, i));
  }
  return e.startsWith("pointer") || e.startsWith("touch") || e === "wheel" ? te(() => {
    t.addEventListener(e, s, r);
  }) : t.addEventListener(e, s, r), s;
}
function Wr(e, t, n, r, s) {
  var i = { capture: r, passive: s }, u = er(e, t, n, i);
  (t === document.body || // @ts-ignore
  t === window || // @ts-ignore
  t === document || // Firefox has quirky behavior, it can happen that we still get "canplay" events when the element is already removed
  t instanceof HTMLMediaElement) && jt(() => {
    t.removeEventListener(e, u, i);
  });
}
function Xr(e, t, n) {
  (t[ye] ??= {})[e] = n;
}
function Zr(e) {
  for (var t = 0; t < e.length; t++)
    Zt.add(e[t]);
  for (var n of ze)
    n(e);
}
let vt = null;
function Ke(e) {
  var t = this, n = (
    /** @type {Node} */
    t.ownerDocument
  ), r = e.type, s = e.composedPath?.() || [], i = (
    /** @type {null | Element} */
    s[0] || e.target
  );
  vt = e;
  var u = 0, a = vt === e && e[ye];
  if (a) {
    var l = s.indexOf(a);
    if (l !== -1 && (t === document || t === /** @type {any} */
    window)) {
      e[ye] = t;
      return;
    }
    var f = s.indexOf(t);
    if (f === -1)
      return;
    l <= f && (u = l);
  }
  if (i = /** @type {Element} */
  s[u] || e.target, i !== t) {
    nn(e, "currentTarget", {
      configurable: !0,
      get() {
        return i || n;
      }
    });
    var _ = v, o = p;
    O(null), I(null);
    try {
      for (var c, h = []; i !== null && i !== t; ) {
        try {
          var d = i[ye]?.[r];
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
      e[ye] = t, delete e.currentTarget, O(_), I(o);
    }
  }
}
function Jr(e) {
  return e.endsWith("capture") && e !== "gotpointercapture" && e !== "lostpointercapture";
}
const tr = [
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
function Qr(e) {
  return tr.includes(e);
}
const nr = {
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
function es(e) {
  return e = e.toLowerCase(), nr[e] ?? e;
}
const rr = ["touchstart", "touchmove"];
function sr(e) {
  return rr.includes(e);
}
function ts(e, t) {
  var n = t == null ? "" : typeof t == "object" ? `${t}` : t;
  n !== /** @type {any} */
  (e[Ue] ??= e.nodeValue) && (e[Ue] = n, e.nodeValue = `${n}`);
}
function ns(e, t) {
  return ir(e, t);
}
const ke = /* @__PURE__ */ new Map();
function ir(e, { target: t, anchor: n, props: r = {}, events: s, context: i, intro: u = !0, transformError: a }) {
  Yn();
  var l = void 0, f = $n(() => {
    var _ = n ?? t.appendChild(Dt());
    Pn(
      /** @type {TemplateNode} */
      _,
      {
        pending: () => {
        }
      },
      (h) => {
        kn({});
        var d = (
          /** @type {ComponentContext} */
          E
        );
        i && (d.c = i), s && (r.$$events = s), l = e(h, r) || {}, xn();
      },
      a
    );
    var o = /* @__PURE__ */ new Set(), c = (h) => {
      for (var d = 0; d < h.length; d++) {
        var y = h[d];
        if (!o.has(y)) {
          o.add(y);
          var U = sr(y);
          for (const Be of [t, document]) {
            var V = ke.get(Be);
            V === void 0 && (V = /* @__PURE__ */ new Map(), ke.set(Be, V));
            var le = V.get(y);
            le === void 0 ? (Be.addEventListener(y, Ke, { passive: U }), V.set(y, 1)) : V.set(y, le + 1);
          }
        }
      }
    };
    return c(tn(Zt)), ze.add(c), () => {
      for (var h of o)
        for (const U of [t, document]) {
          var d = (
            /** @type {Map<string, number>} */
            ke.get(U)
          ), y = (
            /** @type {number} */
            d.get(h)
          );
          --y == 0 ? (U.removeEventListener(h, Ke), d.delete(h), d.size === 0 && ke.delete(U)) : d.set(h, y);
        }
      ze.delete(c), _ !== n && _.parentNode?.removeChild(_);
    };
  });
  return We.set(l, f), l;
}
let We = /* @__PURE__ */ new WeakMap();
function rs(e, t) {
  const n = We.get(e);
  return n ? (We.delete(e), n(t)) : Promise.resolve();
}
export {
  Sr as $,
  De as A,
  kr as B,
  ar as C,
  Ln as D,
  me as E,
  tn as F,
  fr as G,
  Ae as H,
  lr as I,
  or as J,
  Ir as K,
  mr as L,
  D as M,
  R as N,
  H as O,
  ur as P,
  Lr as Q,
  je as R,
  jt as S,
  pr as T,
  yr as U,
  $r as V,
  qr as W,
  cn as X,
  hn as Y,
  Cr as Z,
  Nr as _,
  Dt as a,
  on as a0,
  Jt as a1,
  Ar as a2,
  rn as a3,
  Mn as a4,
  wr as a5,
  Jr as a6,
  Xr as a7,
  Zr as a8,
  er as a9,
  xn as aA,
  kn as aB,
  Fr as aC,
  Wr as aD,
  Gr as aE,
  Pr as aF,
  ht as aG,
  Mr as aH,
  Or as aI,
  ts as aJ,
  ns as aK,
  rs as aL,
  es as aa,
  b as ab,
  Qr as ac,
  st as ad,
  at as ae,
  ee as af,
  Vr as ag,
  Er as ah,
  Kr as ai,
  Je as aj,
  we as ak,
  xr as al,
  _r as am,
  ge as an,
  G as ao,
  dr as ap,
  hr as aq,
  cr as ar,
  vr as as,
  Z as at,
  br as au,
  Y as av,
  Rr as aw,
  Ur as ax,
  Yr as ay,
  Dr as az,
  p as b,
  Br as c,
  Ft as d,
  Un as e,
  gr as f,
  gt as g,
  zr as h,
  Qt as i,
  j,
  J as k,
  g as l,
  Zn as m,
  fn as n,
  sn as o,
  Ce as p,
  te as q,
  an as r,
  Tn as s,
  jr as t,
  lt as u,
  E as v,
  Le as w,
  Hr as x,
  Kn as y,
  z
};
