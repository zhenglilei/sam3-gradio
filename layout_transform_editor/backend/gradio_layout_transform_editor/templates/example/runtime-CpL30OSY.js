let Fe = !1, Pt = !1;
function Tn() {
  Fe = !0;
}
const An = 2, Rn = 4, On = 8, Cn = 2, g = /* @__PURE__ */ Symbol("uninitialized"), Pn = "http://www.w3.org/1999/xhtml", Ze = !1;
var Dt = Array.isArray, Nt = Array.prototype.indexOf, we = Array.prototype.includes, Dn = Array.from, Nn = Object.defineProperty, fe = Object.getOwnPropertyDescriptor, Ft = Object.getOwnPropertyDescriptors, jt = Object.prototype, It = Array.prototype, Je = Object.getPrototypeOf, He = Object.isExtensible;
const Mt = () => {
};
function Fn(e) {
  return e();
}
function Lt(e) {
  for (var t = 0; t < e.length; t++)
    e[t]();
}
function Qe() {
  var e, t, n = new Promise((r, s) => {
    e = r, t = s;
  });
  return { promise: n, resolve: e, reject: t };
}
const b = 2, ae = 4, ce = 8, et = 1 << 24, O = 16, F = 32, Y = 64, qt = 128, A = 512, m = 1024, E = 2048, j = 4096, R = 8192, D = 16384, re = 32768, ze = 1 << 25, ye = 65536, ge = 1 << 17, Ut = 1 << 18, _e = 1 << 19, tt = 1 << 20, $ = 65536, me = 1 << 21, ee = 1 << 22, U = 1 << 23, ue = /* @__PURE__ */ Symbol("$state"), jn = /* @__PURE__ */ Symbol("legacy props"), Yt = /* @__PURE__ */ Symbol("attributes"), Bt = /* @__PURE__ */ Symbol("class"), Ht = /* @__PURE__ */ Symbol("style"), zt = /* @__PURE__ */ Symbol("text"), ke = new class extends Error {
  name = "StaleReactionError";
  message = "The reaction that called `getAbortSignal()` was re-run or destroyed";
}();
function Gt() {
  throw new Error("https://svelte.dev/e/async_derived_orphan");
}
function Kt(e) {
  throw new Error("https://svelte.dev/e/effect_in_teardown");
}
function Vt() {
  throw new Error("https://svelte.dev/e/effect_in_unowned_derived");
}
function $t(e) {
  throw new Error("https://svelte.dev/e/effect_orphan");
}
function Wt() {
  throw new Error("https://svelte.dev/e/effect_update_depth_exceeded");
}
function Mn(e) {
  throw new Error("https://svelte.dev/e/props_invalid_value");
}
function Xt() {
  throw new Error("https://svelte.dev/e/state_descriptors_fixed");
}
function Zt() {
  throw new Error("https://svelte.dev/e/state_prototype_fixed");
}
function Jt() {
  throw new Error("https://svelte.dev/e/state_unsafe_mutation");
}
function Ln() {
  throw new Error("https://svelte.dev/e/svelte_boundary_reset_onerror");
}
function Qt() {
  console.warn("https://svelte.dev/e/derived_inert");
}
function qn() {
  console.warn("https://svelte.dev/e/svelte_boundary_reset_noop");
}
function nt(e) {
  return e === this.v;
}
function en(e, t) {
  return e != e ? t == t : e !== t || e !== null && typeof e == "object" || typeof e == "function";
}
function tn(e) {
  return !en(e, this.v);
}
let S = null;
function Ee(e) {
  S = e;
}
function Un(e, t = !1, n) {
  S = {
    p: S,
    i: !1,
    c: null,
    e: null,
    s: e,
    x: null,
    r: (
      /** @type {Effect} */
      p
    ),
    l: Fe && !t ? { s: null, u: null, $: [] } : null
  };
}
function Yn(e) {
  var t = (
    /** @type {ComponentContext} */
    S
  ), n = t.e;
  if (n !== null) {
    t.e = null;
    for (var r of n)
      mt(r);
  }
  return t.i = !0, S = t.p, /** @type {T} */
  {};
}
function ve() {
  return !Fe || S !== null && S.l === null;
}
let J = [];
function nn() {
  var e = J;
  J = [], Lt(e);
}
function Ge(e) {
  if (J.length === 0) {
    var t = J;
    queueMicrotask(() => {
      t === J && nn();
    });
  }
  J.push(e);
}
function rn(e) {
  var t = p;
  if (t === null)
    return h.f |= U, e;
  if ((t.f & re) === 0 && (t.f & ae) === 0)
    throw e;
  be(e, t);
}
function be(e, t) {
  if (!(t !== null && (t.f & D) !== 0)) {
    for (; t !== null; ) {
      if ((t.f & qt) !== 0) {
        if ((t.f & re) === 0)
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
const sn = -7169;
function y(e, t) {
  e.f = e.f & sn | t;
}
function je(e) {
  (e.f & A) !== 0 || e.deps === null ? y(e, m) : y(e, j);
}
function rt(e) {
  if (e !== null)
    for (const t of e)
      (t.f & b) === 0 || (t.f & $) === 0 || (t.f ^= $, rt(
        /** @type {Derived} */
        t.deps
      ));
}
function ln(e, t, n) {
  (e.f & E) !== 0 ? t.add(e) : (e.f & j) !== 0 && n.add(e), rt(e.deps), y(e, m);
}
function fn(e, t, n, r) {
  const s = ve() ? it : on;
  var l = e.filter((_) => !_.settled), a = t.map(s);
  if (n.length === 0 && l.length === 0) {
    r(a);
    return;
  }
  var f = (
    /** @type {Effect} */
    p
  ), i = un(), u = l.length === 1 ? l[0].promise : l.length > 1 ? Promise.all(l.map((_) => _.promise)) : null;
  function v(_) {
    if ((f.f & D) === 0) {
      i();
      try {
        r([...a, ..._]);
      } catch (d) {
        be(d, f);
      }
      xe();
    }
  }
  var o = st();
  if (n.length === 0) {
    u.then(() => v([])).finally(o);
    return;
  }
  function c() {
    Promise.all(n.map((_) => /* @__PURE__ */ an(_))).then(v).catch((_) => be(_, f)).finally(o);
  }
  u ? u.then(() => {
    i(), c(), xe();
  }) : c();
}
function un() {
  var e = (
    /** @type {Effect} */
    p
  ), t = h, n = S, r = (
    /** @type {Batch} */
    w
  );
  return function(l = !0) {
    te(e), H(t), Ee(n), l && (e.f & D) === 0 && (r?.activate(), r?.apply());
  };
}
function xe(e = !0) {
  te(null), H(null), Ee(null), e && w?.deactivate();
}
function st() {
  var e = (
    /** @type {Effect} */
    p
  ), t = e.b, n = (
    /** @type {Batch} */
    w
  ), r = !!t?.is_rendered();
  return t?.update_pending_count(1, n), n.increment(r, e), () => {
    t?.update_pending_count(-1, n), n.decrement(r, e);
  };
}
// @__NO_SIDE_EFFECTS__
function it(e) {
  var t = b | E;
  return p !== null && (p.f |= _e), {
    ctx: S,
    deps: null,
    effects: null,
    equals: nt,
    f: t,
    fn: e,
    reactions: null,
    rv: 0,
    v: (
      /** @type {V} */
      g
    ),
    wv: 0,
    parent: p,
    ac: null
  };
}
const se = /* @__PURE__ */ Symbol("obsolete");
// @__NO_SIDE_EFFECTS__
function an(e, t, n) {
  let r = (
    /** @type {Effect | null} */
    p
  );
  r === null && Gt();
  var s = (
    /** @type {Promise<V>} */
    /** @type {unknown} */
    void 0
  ), l = _t(
    /** @type {V} */
    g
  ), a = !h, f = /* @__PURE__ */ new Set();
  return gn(() => {
    var i = (
      /** @type {Effect} */
      p
    ), u = Qe();
    s = u.promise;
    try {
      Promise.resolve(e()).then(u.resolve, (_) => {
        _ !== ke && u.reject(_);
      }).finally(xe);
    } catch (_) {
      u.reject(_), xe();
    }
    var v = (
      /** @type {Batch} */
      w
    );
    if (a) {
      if ((i.f & re) !== 0)
        var o = st();
      if (
        // boundary can be null if the async derived is inside an $effect.root not connected to the component render tree
        r.b?.is_rendered()
      )
        v.async_deriveds.get(i)?.reject(se);
      else
        for (const _ of f.values())
          _.reject(se);
      f.add(u), v.async_deriveds.set(i, u);
    }
    const c = (_, d = void 0) => {
      o?.(), f.delete(u), d !== se && (v.activate(), d ? (l.f |= U, De(l, d)) : ((l.f & U) !== 0 && (l.f ^= U), De(l, _)), v.deactivate());
    };
    u.promise.then(c, (_) => c(null, _ || "unknown"));
  }), yn(() => {
    for (const i of f)
      i.reject(se);
  }), new Promise((i) => {
    function u(v) {
      function o() {
        v === s ? i(l) : u(s);
      }
      v.then(o, o);
    }
    u(s);
  });
}
// @__NO_SIDE_EFFECTS__
function on(e) {
  const t = /* @__PURE__ */ it(e);
  return t.equals = tn, t;
}
function cn(e) {
  var t = e.effects;
  if (t !== null) {
    e.effects = null;
    for (var n = 0; n < t.length; n += 1)
      X(
        /** @type {Effect} */
        t[n]
      );
  }
}
function Ie(e) {
  var t, n = p, r = e.parent;
  if (!B && r !== null && e.v !== g && // if it was never evaluated before, it's guaranteed to fail downstream, so we try to execute instead
  (r.f & (D | R)) !== 0)
    return Qt(), e.v;
  te(r);
  try {
    e.f &= ~$, cn(e), t = Rt(e);
  } finally {
    te(n);
  }
  return t;
}
function lt(e) {
  var t = Ie(e);
  if (!e.equals(t) && (e.wv = Tt(), (!w?.is_fork || e.deps === null) && (w !== null ? (w.capture(e, t, !0), Ce?.capture(e, t, !0)) : e.v = t, e.deps === null))) {
    y(e, m);
    return;
  }
  B || (C !== null ? (gt() || w?.is_fork) && C.set(e, t) : je(e));
}
function _n(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      (t.teardown || t.ac) && (t.teardown?.(), t.ac?.abort(ke), t.fn !== null && (t.teardown = Mt), t.ac = null, oe(t, 0), Ue(t));
}
function ft(e) {
  if (e.effects !== null)
    for (const t of e.effects)
      t.teardown && t.fn !== null && ne(t);
}
let Ae = null, Z = null, w = null, Ce = null, C = null, Pe = null, Re = !1, Q = null, he = null;
var Ke = 0;
let vn = 1;
class W {
  id = vn++;
  /** True as soon as `#process` was called */
  #c = !1;
  linked = !0;
  /** @type {Batch | null} */
  #i = null;
  /** @type {Batch | null} */
  #f = null;
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
  #_ = /* @__PURE__ */ new Set();
  /**
   * If a fork is discarded, we need to destroy any effects that are no longer needed
   * @type {Set<(batch: Batch) => void>}
   */
  #v = /* @__PURE__ */ new Set();
  /**
   * The number of async effects that are currently in flight
   */
  #d = 0;
  /**
   * Async effects that are currently in flight, _not_ inside a pending boundary
   * @type {Map<Effect, number>}
   */
  #r = /* @__PURE__ */ new Map();
  /**
   * A deferred that resolves when the batch is committed, used with `settled()`
   * TODO replace with Promise.withResolvers once supported widely enough
   * @type {{ promise: Promise<void>, resolve: (value?: any) => void, reject: (reason: unknown) => void } | null}
   */
  #h = null;
  /**
   * The root effects that need to be flushed
   * @type {Effect[]}
   */
  #e = [];
  /**
   * Effects created while this batch was active.
   * @type {Effect[]}
   */
  #w = [];
  /**
   * Deferred effects (which run after async work has completed) that are DIRTY
   * @type {Set<Effect>}
   */
  #s = /* @__PURE__ */ new Set();
  /**
   * Deferred effects that are MAYBE_DIRTY
   * @type {Set<Effect>}
   */
  #t = /* @__PURE__ */ new Set();
  /**
   * A map of branches that still exist, but will be destroyed when this batch
   * is committed — we skip over these during `process`.
   * The value contains child effects that were dirty/maybe_dirty before being reset,
   * so they can be rescheduled if the branch survives.
   * @type {Map<Effect, { d: Effect[], m: Effect[] }>}
   */
  #n = /* @__PURE__ */ new Map();
  /**
   * Inverse of #skipped_branches which we need to tell prior batches to unskip them when committing
   * @type {Set<Effect>}
   */
  #p = /* @__PURE__ */ new Set();
  is_fork = !1;
  #u = !1;
  constructor() {
    Z === null ? Ae = Z = this : (Z.#f = this, this.#i = Z), Z = this;
  }
  #y() {
    if (this.is_fork) return !0;
    for (const r of this.#r.keys()) {
      for (var t = r, n = !1; t.parent !== null; ) {
        if (this.#n.has(t)) {
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
    this.#n.has(t) || this.#n.set(t, { d: [], m: [] }), this.#p.delete(t);
  }
  /**
   * Remove an effect from the #skipped_branches map and reschedule
   * any tracked dirty/maybe_dirty child effects
   * @param {Effect} effect
   * @param {(e: Effect) => void} callback
   */
  unskip_effect(t, n = (r) => this.schedule(r)) {
    var r = this.#n.get(t);
    if (r) {
      this.#n.delete(t);
      for (var s of r.d)
        y(s, E), n(s);
      for (s of r.m)
        y(s, j), n(s);
    }
    this.#p.add(t);
  }
  #a() {
    this.#c = !0, Ke++ > 1e3 && (this.#o(), dn());
    for (const i of this.#s)
      this.#t.delete(i), y(i, E), this.schedule(i);
    for (const i of this.#t)
      y(i, j), this.schedule(i);
    const t = this.#e;
    this.#e = [], this.apply();
    var n = Q = [], r = [], s = he = [];
    for (const i of t)
      try {
        this.#g(i, n, r);
      } catch (u) {
        throw ot(i), this.#y() || this.discard(), u;
      }
    if (w = null, s.length > 0) {
      var l = W.ensure();
      for (const i of s)
        l.schedule(i);
    }
    if (Q = null, he = null, this.#y()) {
      this.#l(r), this.#l(n);
      for (const [i, u] of this.#n)
        at(i, u);
      s.length > 0 && /** @type {unknown} */
      w.#a();
      return;
    }
    const a = this.#m();
    if (a) {
      this.#l(r), this.#l(n), a.#E(this);
      return;
    }
    this.#s.clear(), this.#t.clear();
    for (const i of this.#_) i(this);
    this.#_.clear(), Ce = this, Ve(r), Ve(n), Ce = null, this.#h?.resolve();
    var f = (
      /** @type {Batch | null} */
      /** @type {unknown} */
      w
    );
    if (this.#d === 0 && (this.#e.length === 0 || f !== null) && this.#o(), this.#e.length > 0)
      if (f !== null) {
        const i = f;
        i.#e.push(...this.#e.filter((u) => !i.#e.includes(u)));
      } else
        f = this;
    f !== null && f.#a();
  }
  /**
   * Traverse the effect tree, executing effects or stashing
   * them for later execution as appropriate
   * @param {Effect} root
   * @param {Effect[]} effects
   * @param {Effect[]} render_effects
   */
  #g(t, n, r) {
    t.f ^= m;
    for (var s = t.first; s !== null; ) {
      var l = s.f, a = (l & (F | Y)) !== 0, f = a && (l & m) !== 0, i = f || (l & R) !== 0 || this.#n.has(s);
      if (!i && s.fn !== null) {
        a ? s.f ^= m : (l & ae) !== 0 ? n.push(s) : de(s) && ((l & O) !== 0 && this.#t.add(s), ne(s));
        var u = s.first;
        if (u !== null) {
          s = u;
          continue;
        }
      }
      for (; s !== null; ) {
        var v = s.next;
        if (v !== null) {
          s = v;
          break;
        }
        s = s.parent;
      }
    }
  }
  #m() {
    for (var t = this.#i; t !== null; ) {
      if (!t.is_fork) {
        for (const [n, [, r]] of this.current)
          if (t.current.has(n) && !r)
            return t;
      }
      t = t.#i;
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
      const l = this.async_deriveds.get(r);
      l && s.promise.then(l.resolve).catch(l.reject);
    }
    t.async_deriveds.clear(), this.transfer_effects(t.#s, t.#t);
    const n = (r) => {
      var s = r.reactions;
      if (s !== null)
        for (const f of s) {
          var l = f.f;
          if ((l & b) !== 0)
            n(
              /** @type {Derived} */
              f
            );
          else {
            var a = (
              /** @type {Effect} */
              f
            );
            l & (ee | O) && !this.async_deriveds.has(a) && (this.#t.delete(a), y(a, E), this.schedule(a));
          }
        }
    };
    for (const r of this.current.keys())
      n(r);
    this.oncommit(() => t.discard()), t.#o(), w = this, this.#a();
  }
  /**
   * @param {Effect[]} effects
   */
  #l(t) {
    for (var n = 0; n < t.length; n += 1)
      ln(t[n], this.#s, this.#t);
  }
  /**
   * Associate a change to a given source with the current
   * batch, noting its previous and current values
   * @param {Value} source
   * @param {any} value
   * @param {boolean} [is_derived]
   */
  capture(t, n, r = !1) {
    t.v !== g && !this.previous.has(t) && this.previous.set(t, t.v), (t.f & U) === 0 && (this.current.set(t, [n, r]), C?.set(t, n)), this.is_fork || (t.v = n);
  }
  activate() {
    w = this;
  }
  deactivate() {
    w = null, C = null;
  }
  flush() {
    try {
      Re = !0, w = this, this.#a();
    } finally {
      Ke = 0, Pe = null, Q = null, he = null, Re = !1, w = null, C = null, K.clear();
    }
  }
  discard() {
    for (const t of this.#v) t(this);
    this.#v.clear();
    for (const t of this.async_deriveds.values())
      t.reject(se);
    this.#o(), this.#h?.resolve();
  }
  /**
   * @param {Effect} effect
   */
  register_created_effect(t) {
    this.#w.push(t);
  }
  #b() {
    for (let o = Ae; o !== null; o = o.#f) {
      var t = o.id < this.id, n = [];
      for (const [c, [_, d]] of this.current) {
        if (o.current.has(c)) {
          var r = (
            /** @type {[any, boolean]} */
            o.current.get(c)[0]
          );
          if (t && _ !== r)
            o.current.set(c, [_, d]);
          else
            continue;
        }
        n.push(c);
      }
      if (t)
        for (const [c, _] of this.async_deriveds) {
          const d = o.async_deriveds.get(c);
          d && _.promise.then(d.resolve).catch(d.reject);
        }
      var s = [...o.current.keys()].filter(
        (c) => !/** @type {[any, boolean]} */
        o.current.get(c)[1]
      );
      if (!(!o.#c || s.length === 0)) {
        var l = s.filter((c) => !this.current.has(c));
        if (l.length === 0)
          t && o.discard();
        else if (n.length > 0) {
          if (t)
            for (const c of this.#p)
              o.unskip_effect(c, (_) => {
                (_.f & (O | ee)) !== 0 ? o.schedule(_) : o.#l([_]);
              });
          o.activate();
          var a = /* @__PURE__ */ new Set(), f = /* @__PURE__ */ new Map();
          for (var i of n)
            ut(i, l, a, f);
          f = /* @__PURE__ */ new Map();
          var u = [...o.current].filter(([c, _]) => {
            const d = this.current.get(c);
            return d ? d[0] !== _[0] || d[1] !== _[1] : !0;
          }).map(([c]) => c);
          if (u.length > 0)
            for (const c of this.#w)
              (c.f & (D | R | ge)) === 0 && Me(c, u, f) && ((c.f & (ee | O)) !== 0 ? (y(c, E), o.schedule(c)) : o.#s.add(c));
          if (o.#e.length > 0 && !o.#u) {
            o.apply();
            for (var v of o.#e)
              o.#g(v, [], []);
            o.#e = [];
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
    if (this.#d += 1, t) {
      let r = this.#r.get(n) ?? 0;
      this.#r.set(n, r + 1);
    }
  }
  /**
   * @param {boolean} blocking
   * @param {Effect} effect
   */
  decrement(t, n) {
    if (this.#d -= 1, t) {
      let r = this.#r.get(n) ?? 0;
      r === 1 ? this.#r.delete(n) : this.#r.set(n, r - 1);
    }
    this.#u || (this.#u = !0, Ge(() => {
      this.#u = !1, this.linked && this.flush();
    }));
  }
  /**
   * @param {Set<Effect>} dirty_effects
   * @param {Set<Effect>} maybe_dirty_effects
   */
  transfer_effects(t, n) {
    for (const r of t)
      this.#s.add(r);
    for (const r of n)
      this.#t.add(r);
    t.clear(), n.clear();
  }
  /** @param {(batch: Batch) => void} fn */
  oncommit(t) {
    this.#_.add(t);
  }
  /** @param {(batch: Batch) => void} fn */
  ondiscard(t) {
    this.#v.add(t);
  }
  settled() {
    return (this.#h ??= Qe()).promise;
  }
  static ensure() {
    if (w === null) {
      const t = w = new W();
      Re || Ge(() => {
        t.#c || t.flush();
      });
    }
    return w;
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
    if (Pe = t, t.b?.is_pending && (t.f & (ae | ce | et)) !== 0 && (t.f & re) === 0) {
      t.b.defer_effect(t);
      return;
    }
    for (var n = t; n.parent !== null; ) {
      n = n.parent;
      var r = n.f;
      if (Q !== null && n === p && (h === null || (h.f & b) === 0))
        return;
      if ((r & (Y | F)) !== 0) {
        if ((r & m) === 0)
          return;
        n.f ^= m;
      }
    }
    this.#e.push(n);
  }
  #o() {
    if (this.linked) {
      var t = this.#i, n = this.#f;
      t === null ? Ae = n : t.#f = n, n === null ? Z = t : n.#i = t, this.linked = !1;
    }
  }
}
function dn() {
  try {
    Wt();
  } catch (e) {
    be(e, Pe);
  }
}
let M = null;
function Ve(e) {
  var t = e.length;
  if (t !== 0) {
    for (var n = 0; n < t; ) {
      var r = e[n++];
      if ((r.f & (D | R)) === 0 && de(r) && (M = /* @__PURE__ */ new Set(), ne(r), r.deps === null && r.first === null && r.nodes === null && r.teardown === null && r.ac === null && bt(r), M?.size > 0)) {
        K.clear();
        for (const s of M) {
          if ((s.f & (D | R)) !== 0) continue;
          const l = [s];
          let a = s.parent;
          for (; a !== null; )
            M.has(a) && (M.delete(a), l.push(a)), a = a.parent;
          for (let f = l.length - 1; f >= 0; f--) {
            const i = l[f];
            (i.f & (D | R)) === 0 && ne(i);
          }
        }
        M.clear();
      }
    }
    M = null;
  }
}
function ut(e, t, n, r) {
  if (!n.has(e) && (n.add(e), e.reactions !== null))
    for (const s of e.reactions) {
      const l = s.f;
      (l & b) !== 0 ? ut(
        /** @type {Derived} */
        s,
        t,
        n,
        r
      ) : (l & (ee | O)) !== 0 && (l & E) === 0 && Me(s, t, r) && (y(s, E), Le(
        /** @type {Effect} */
        s
      ));
    }
}
function Me(e, t, n) {
  const r = n.get(e);
  if (r !== void 0) return r;
  if (e.deps !== null)
    for (const s of e.deps) {
      if (we.call(t, s))
        return !0;
      if ((s.f & b) !== 0 && Me(
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
function Le(e) {
  w.schedule(e);
}
function at(e, t) {
  if (!((e.f & F) !== 0 && (e.f & m) !== 0)) {
    (e.f & E) !== 0 ? t.d.push(e) : (e.f & j) !== 0 && t.m.push(e), y(e, m);
    for (var n = e.first; n !== null; )
      at(n, t), n = n.next;
  }
}
function ot(e) {
  y(e, m);
  for (var t = e.first; t !== null; )
    ot(t), t = t.next;
}
let Se = /* @__PURE__ */ new Set();
const K = /* @__PURE__ */ new Map();
let ct = !1;
function _t(e, t) {
  var n = {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v: e,
    reactions: null,
    equals: nt,
    rv: 0,
    wv: 0
  };
  return n;
}
// @__NO_SIDE_EFFECTS__
function q(e, t) {
  const n = _t(e);
  return xn(n), n;
}
function z(e, t, n = !1) {
  h !== null && // since we are untracking the function inside `$inspect.with` we need to add this check
  // to ensure we error if state is set inside an inspect effect
  (!P || (h.f & ge) !== 0) && ve() && (h.f & (b | O | ee | ge)) !== 0 && (N === null || !N.has(e)) && Jt();
  let r = n ? ie(t) : t;
  return De(e, r, he);
}
function De(e, t, n = null) {
  if (!e.equals(t)) {
    K.set(e, B ? t : e.v);
    var r = W.ensure();
    if (r.capture(e, t), (e.f & b) !== 0) {
      const s = (
        /** @type {Derived} */
        e
      );
      (e.f & E) !== 0 && Ie(s), C === null && je(s);
    }
    e.wv = Tt(), vt(e, E, n), ve() && p !== null && (p.f & m) !== 0 && (p.f & (F | Y)) === 0 && (T === null ? Sn([e]) : T.push(e)), !r.is_fork && Se.size > 0 && !ct && hn();
  }
  return t;
}
function hn() {
  ct = !1;
  for (const e of Se) {
    (e.f & m) !== 0 && y(e, j);
    let t;
    try {
      t = de(e);
    } catch {
      t = !0;
    }
    t && ne(e);
  }
  Se.clear();
}
function Oe(e) {
  z(e, e.v + 1);
}
function vt(e, t, n) {
  var r = e.reactions;
  if (r !== null)
    for (var s = ve(), l = r.length, a = 0; a < l; a++) {
      var f = r[a], i = f.f;
      if (!(!s && f === p)) {
        var u = (i & E) === 0;
        if (u && y(f, t), (i & ge) !== 0)
          Se.add(
            /** @type {Effect} */
            f
          );
        else if ((i & b) !== 0) {
          var v = (
            /** @type {Derived} */
            f
          );
          C?.delete(v), (i & $) === 0 && (i & A && (p === null || (p.f & me) === 0) && (f.f |= $), vt(v, j, n));
        } else if (u) {
          var o = (
            /** @type {Effect} */
            f
          );
          (i & O) !== 0 && M !== null && M.add(o), n !== null ? n.push(o) : Le(o);
        }
      }
    }
}
function ie(e) {
  if (typeof e != "object" || e === null || ue in e)
    return e;
  const t = Je(e);
  if (t !== jt && t !== It)
    return e;
  var n = /* @__PURE__ */ new Map(), r = Dt(e), s = /* @__PURE__ */ q(0), l = V, a = (f) => {
    if (V === l)
      return f();
    var i = h, u = V;
    H(null), Xe(l);
    var v = f();
    return H(i), Xe(u), v;
  };
  return r && n.set("length", /* @__PURE__ */ q(
    /** @type {any[]} */
    e.length
  )), new Proxy(
    /** @type {any} */
    e,
    {
      defineProperty(f, i, u) {
        (!("value" in u) || u.configurable === !1 || u.enumerable === !1 || u.writable === !1) && Xt();
        var v = n.get(i);
        return v === void 0 ? a(() => {
          var o = /* @__PURE__ */ q(u.value);
          return n.set(i, o), o;
        }) : z(v, u.value, !0), !0;
      },
      deleteProperty(f, i) {
        var u = n.get(i);
        if (u === void 0) {
          if (i in f) {
            const v = a(() => /* @__PURE__ */ q(g));
            n.set(i, v), Oe(s);
          }
        } else
          z(u, g), Oe(s);
        return !0;
      },
      get(f, i, u) {
        if (i === ue)
          return e;
        var v = n.get(i), o = i in f;
        if (v === void 0 && (!o || fe(f, i)?.writable) && (v = a(() => {
          var _ = ie(o ? f[i] : g), d = /* @__PURE__ */ q(_);
          return d;
        }), n.set(i, v)), v !== void 0) {
          var c = le(v);
          return c === g ? void 0 : c;
        }
        return Reflect.get(f, i, u);
      },
      getOwnPropertyDescriptor(f, i) {
        var u = Reflect.getOwnPropertyDescriptor(f, i);
        if (u && "value" in u) {
          var v = n.get(i);
          v && (u.value = le(v));
        } else if (u === void 0) {
          var o = n.get(i), c = o?.v;
          if (o !== void 0 && c !== g)
            return {
              enumerable: !0,
              configurable: !0,
              value: c,
              writable: !0
            };
        }
        return u;
      },
      has(f, i) {
        if (i === ue)
          return !0;
        var u = n.get(i), v = u !== void 0 && u.v !== g || Reflect.has(f, i);
        if (u !== void 0 || p !== null && (!v || fe(f, i)?.writable)) {
          u === void 0 && (u = a(() => {
            var c = v ? ie(f[i]) : g, _ = /* @__PURE__ */ q(c);
            return _;
          }), n.set(i, u));
          var o = le(u);
          if (o === g)
            return !1;
        }
        return v;
      },
      set(f, i, u, v) {
        var o = n.get(i), c = i in f;
        if (r && i === "length")
          for (var _ = u; _ < /** @type {Source<number>} */
          o.v; _ += 1) {
            var d = n.get(_ + "");
            d !== void 0 ? z(d, g) : _ in f && (d = a(() => /* @__PURE__ */ q(g)), n.set(_ + "", d));
          }
        if (o === void 0)
          (!c || fe(f, i)?.writable) && (o = a(() => /* @__PURE__ */ q(void 0)), z(o, ie(u)), n.set(i, o));
        else {
          c = o.v !== g;
          var I = a(() => ie(u));
          z(o, I);
        }
        var Ye = Reflect.getOwnPropertyDescriptor(f, i);
        if (Ye?.set && Ye.set.call(v, u), !c) {
          if (r && typeof i == "string") {
            var Be = (
              /** @type {Source<number>} */
              n.get("length")
            ), Te = Number(i);
            Number.isInteger(Te) && Te >= Be.v && z(Be, Te + 1);
          }
          Oe(s);
        }
        return !0;
      },
      ownKeys(f) {
        le(s);
        var i = Reflect.ownKeys(f).filter((o) => {
          var c = n.get(o);
          return c === void 0 || c.v !== g;
        });
        for (var [u, v] of n)
          v.v !== g && !(u in f) && i.push(u);
        return i;
      },
      setPrototypeOf() {
        Zt();
      }
    }
  );
}
var $e, pn, dt, ht;
function Bn() {
  if ($e === void 0) {
    $e = window, pn = /Firefox/.test(navigator.userAgent);
    var e = Element.prototype, t = Node.prototype, n = Text.prototype;
    dt = fe(t, "firstChild").get, ht = fe(t, "nextSibling").get, He(e) && (e[Bt] = void 0, e[Yt] = null, e[Ht] = void 0, e.__e = void 0), He(n) && (n[zt] = void 0);
  }
}
function Hn(e = "") {
  return document.createTextNode(e);
}
// @__NO_SIDE_EFFECTS__
function pt(e) {
  return (
    /** @type {TemplateNode | null} */
    dt.call(e)
  );
}
// @__NO_SIDE_EFFECTS__
function qe(e) {
  return (
    /** @type {TemplateNode | null} */
    ht.call(e)
  );
}
function zn(e, t) {
  return /* @__PURE__ */ pt(e);
}
function Gn(e, t = !1) {
  {
    var n = /* @__PURE__ */ pt(e);
    return n instanceof Comment && n.data === "" ? /* @__PURE__ */ qe(n) : n;
  }
}
function Kn() {
  return !1;
}
function Vn(e, t, n) {
  return (
    /** @type {T extends keyof HTMLElementTagNameMap ? HTMLElementTagNameMap[T] : Element} */
    n ? document.createElement(e, { is: n }) : document.createElement(e)
  );
}
function wt(e) {
  var t = h, n = p;
  H(null), te(null);
  try {
    return e();
  } finally {
    H(t), te(n);
  }
}
function yt(e) {
  p === null && (h === null && $t(), Vt()), B && Kt();
}
function wn(e, t) {
  var n = t.last;
  n === null ? t.last = t.first = e : (n.next = e, e.prev = n, t.last = e);
}
function L(e, t) {
  var n = p;
  n !== null && (n.f & R) !== 0 && (e |= R);
  var r = {
    ctx: S,
    deps: null,
    nodes: null,
    f: e | E | A,
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
  w?.register_created_effect(r);
  var s = r;
  if ((e & ae) !== 0)
    Q !== null ? Q.push(r) : W.ensure().schedule(r);
  else if (t !== null) {
    try {
      ne(r);
    } catch (a) {
      throw X(r), a;
    }
    s.deps === null && s.teardown === null && s.nodes === null && s.first === s.last && // either `null`, or a singular child
    (s.f & _e) === 0 && (s = s.first, (e & O) !== 0 && (e & ye) !== 0 && s !== null && (s.f |= ye));
  }
  if (s !== null && (s.parent = n, n !== null && wn(s, n), h !== null && (h.f & b) !== 0 && (e & Y) === 0)) {
    var l = (
      /** @type {Derived} */
      h
    );
    (l.effects ??= []).push(s);
  }
  return r;
}
function gt() {
  return h !== null && !P;
}
function yn(e) {
  const t = L(ce, null);
  return y(t, m), t.teardown = e, t;
}
function $n(e) {
  yt();
  var t = (
    /** @type {Effect} */
    p.f
  ), n = !h && (t & F) !== 0 && S !== null && !S.i;
  if (n) {
    var r = (
      /** @type {ComponentContext} */
      S
    );
    (r.e ??= []).push(e);
  } else
    return mt(e);
}
function mt(e) {
  return L(ae | tt, e);
}
function Wn(e) {
  return yt(), L(ce | tt, e);
}
function Xn(e) {
  W.ensure();
  const t = L(Y | _e, e);
  return (n = {}) => new Promise((r) => {
    n.outro ? bn(t, () => {
      X(t), r(void 0);
    }) : (X(t), r(void 0));
  });
}
function gn(e) {
  return L(ee | _e, e);
}
function Zn(e, t = 0) {
  return L(ce | t, e);
}
function Jn(e, t = [], n = [], r = []) {
  fn(r, t, n, (s) => {
    L(ce, () => {
      e(...s.map(le));
    });
  });
}
function Qn(e, t = 0) {
  var n = L(O | t, e);
  return n;
}
function er(e) {
  return L(F | _e, e);
}
function Et(e) {
  var t = e.teardown;
  if (t !== null) {
    const n = B, r = h;
    We(!0), H(null);
    try {
      t.call(null);
    } finally {
      We(n), H(r);
    }
  }
}
function Ue(e, t = !1) {
  var n = e.first;
  for (e.first = e.last = null; n !== null; ) {
    const s = n.ac;
    s !== null && wt(() => {
      s.abort(ke);
    });
    var r = n.next;
    (n.f & Y) !== 0 ? n.parent = null : X(n, t), n = r;
  }
}
function mn(e) {
  for (var t = e.first; t !== null; ) {
    var n = t.next;
    (t.f & F) === 0 && X(t), t = n;
  }
}
function X(e, t = !0) {
  var n = !1;
  (t || (e.f & Ut) !== 0) && e.nodes !== null && e.nodes.end !== null && (En(
    e.nodes.start,
    /** @type {TemplateNode} */
    e.nodes.end
  ), n = !0), e.f |= ze, Ue(e, t && !n), oe(e, 0);
  var r = e.nodes && e.nodes.t;
  if (r !== null)
    for (const l of r)
      l.stop();
  Et(e), e.f ^= ze, e.f |= D;
  var s = e.parent;
  s !== null && s.first !== null && bt(e), e.next = e.prev = e.teardown = e.ctx = e.deps = e.fn = e.nodes = e.ac = e.b = null;
}
function En(e, t) {
  for (; e !== null; ) {
    var n = e === t ? null : /* @__PURE__ */ qe(e);
    e.remove(), e = n;
  }
}
function bt(e) {
  var t = e.parent, n = e.prev, r = e.next;
  n !== null && (n.next = r), r !== null && (r.prev = n), t !== null && (t.first === e && (t.first = r), t.last === e && (t.last = n));
}
function bn(e, t, n = !0) {
  var r = [];
  xt(e, r, !0);
  var s = () => {
    n && X(e), t && t();
  }, l = r.length;
  if (l > 0) {
    var a = () => --l || s();
    for (var f of r)
      f.out(a);
  } else
    s();
}
function xt(e, t, n) {
  if ((e.f & R) === 0) {
    e.f ^= R;
    var r = e.nodes && e.nodes.t;
    if (r !== null)
      for (const f of r)
        (f.is_global || n) && t.push(f);
    for (var s = e.first; s !== null; ) {
      var l = s.next;
      if ((s.f & Y) === 0) {
        var a = (s.f & ye) !== 0 || // If this is a branch effect without a block effect parent,
        // it means the parent block effect was pruned. In that case,
        // transparency information was transferred to the branch effect.
        (s.f & F) !== 0 && (e.f & O) !== 0;
        xt(s, t, a ? n : !1);
      }
      s = l;
    }
  }
}
function tr(e) {
  St(e, !0);
}
function St(e, t) {
  if ((e.f & R) !== 0) {
    e.f ^= R, (e.f & m) === 0 && (y(e, E), W.ensure().schedule(e));
    for (var n = e.first; n !== null; ) {
      var r = n.next, s = (n.f & ye) !== 0 || (n.f & F) !== 0;
      St(n, s ? t : !1), n = r;
    }
    var l = e.nodes && e.nodes.t;
    if (l !== null)
      for (const a of l)
        (a.is_global || t) && a.in();
  }
}
function nr(e, t) {
  if (e.nodes)
    for (var n = e.nodes.start, r = e.nodes.end; n !== null; ) {
      var s = n === r ? null : /* @__PURE__ */ qe(n);
      t.append(n), n = s;
    }
}
let pe = !1, B = !1;
function We(e) {
  B = e;
}
let h = null, P = !1;
function H(e) {
  h = e;
}
let p = null;
function te(e) {
  p = e;
}
let N = null;
function xn(e) {
  h !== null && (N ??= /* @__PURE__ */ new Set()).add(e);
}
let x = null, k = 0, T = null;
function Sn(e) {
  T = e;
}
let kt = 1, G = 0, V = G;
function Xe(e) {
  V = e;
}
function Tt() {
  return ++kt;
}
function de(e) {
  var t = e.f;
  if ((t & E) !== 0)
    return !0;
  if (t & b && (e.f &= ~$), (t & j) !== 0) {
    for (var n = (
      /** @type {Value[]} */
      e.deps
    ), r = n.length, s = 0; s < r; s++) {
      var l = n[s];
      if (de(
        /** @type {Derived} */
        l
      ) && lt(
        /** @type {Derived} */
        l
      ), l.wv > e.wv)
        return !0;
    }
    (t & A) !== 0 && // During time traveling we don't want to reset the status so that
    // traversal of the graph in the other batches still happens
    C === null && y(e, m);
  }
  return !1;
}
function At(e, t, n = !0) {
  var r = e.reactions;
  if (r !== null && !(N !== null && N.has(e)))
    for (var s = 0; s < r.length; s++) {
      var l = r[s];
      (l.f & b) !== 0 ? At(
        /** @type {Derived} */
        l,
        t,
        !1
      ) : t === l && (n ? y(l, E) : (l.f & m) !== 0 && y(l, j), Le(
        /** @type {Effect} */
        l
      ));
    }
}
function Rt(e) {
  var t = x, n = k, r = T, s = h, l = N, a = S, f = P, i = V, u = e.f;
  x = /** @type {null | Value[]} */
  null, k = 0, T = null, h = (u & (F | Y)) === 0 ? e : null, N = null, Ee(e.ctx), P = !1, V = ++G, e.ac !== null && (wt(() => {
    e.ac.abort(ke);
  }), e.ac = null);
  try {
    e.f |= me;
    var v = (
      /** @type {Function} */
      e.fn
    ), o = v();
    e.f |= re;
    var c = e.deps, _ = w?.is_fork;
    if (x !== null) {
      var d;
      if (_ || oe(e, k), c !== null && k > 0)
        for (c.length = k + x.length, d = 0; d < x.length; d++)
          c[k + d] = x[d];
      else
        e.deps = c = x;
      if (gt() && (e.f & A) !== 0)
        for (d = k; d < c.length; d++)
          (c[d].reactions ??= []).push(e);
    } else !_ && c !== null && k < c.length && (oe(e, k), c.length = k);
    if (ve() && T !== null && !P && c !== null && (e.f & (b | j | E)) === 0)
      for (d = 0; d < /** @type {Source[]} */
      T.length; d++)
        At(
          T[d],
          /** @type {Effect} */
          e
        );
    if (s !== null && s !== e) {
      if (G++, s.deps !== null)
        for (let I = 0; I < n; I += 1)
          s.deps[I].rv = G;
      if (t !== null)
        for (const I of t)
          I.rv = G;
      T !== null && (r === null ? r = T : r.push(.../** @type {Source[]} */
      T));
    }
    return (e.f & U) !== 0 && (e.f ^= U), o;
  } catch (I) {
    return rn(I);
  } finally {
    e.f ^= me, x = t, k = n, T = r, h = s, N = l, Ee(a), P = f, V = i;
  }
}
function kn(e, t) {
  let n = t.reactions;
  if (n !== null) {
    var r = Nt.call(n, e);
    if (r !== -1) {
      var s = n.length - 1;
      s === 0 ? n = t.reactions = null : (n[r] = n[s], n.pop());
    }
  }
  if (n === null && (t.f & b) !== 0 && // Destroying a child effect while updating a parent effect can cause a dependency to appear
  // to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
  // allows us to skip the expensive work of disconnecting and immediately reconnecting it
  (x === null || !we.call(x, t))) {
    var l = (
      /** @type {Derived} */
      t
    );
    (l.f & A) !== 0 && (l.f ^= A, l.f &= ~$), l.v !== g && je(l), _n(l), oe(l, 0);
  }
}
function oe(e, t) {
  var n = e.deps;
  if (n !== null)
    for (var r = t; r < n.length; r++)
      kn(e, n[r]);
}
function ne(e) {
  var t = e.f;
  if ((t & D) === 0) {
    y(e, m);
    var n = p, r = pe;
    p = e, pe = !0;
    try {
      (t & (O | et)) !== 0 ? mn(e) : Ue(e), Et(e);
      var s = Rt(e);
      e.teardown = typeof s == "function" ? s : null, e.wv = kt;
      var l;
      Ze && Pt && (e.f & E) !== 0 && e.deps;
    } finally {
      pe = r, p = n;
    }
  }
}
function le(e) {
  var t = e.f, n = (t & b) !== 0;
  if (h !== null && !P) {
    var r = p !== null && (p.f & D) !== 0;
    if (!r && (N === null || !N.has(e))) {
      var s = h.deps;
      if ((h.f & me) !== 0)
        e.rv < G && (e.rv = G, x === null && s !== null && s[k] === e ? k++ : x === null ? x = [e] : x.push(e));
      else {
        h.deps ??= [], we.call(h.deps, e) || h.deps.push(e);
        var l = e.reactions;
        l === null ? e.reactions = [h] : we.call(l, h) || l.push(h);
      }
    }
  }
  if (B && K.has(e))
    return K.get(e);
  if (n) {
    var a = (
      /** @type {Derived} */
      e
    );
    if (B) {
      var f = a.v;
      return ((a.f & m) === 0 && a.reactions !== null || Ct(a)) && (f = Ie(a)), K.set(a, f), f;
    }
    var i = (a.f & A) === 0 && !P && h !== null && (pe || (h.f & A) !== 0), u = (a.f & re) === 0;
    de(a) && (i && (a.f |= A), lt(a)), i && !u && (ft(a), Ot(a));
  }
  if (C?.has(e))
    return C.get(e);
  if ((e.f & U) !== 0)
    throw e.v;
  return e.v;
}
function Ot(e) {
  if (e.f |= A, e.deps !== null)
    for (const t of e.deps)
      (t.reactions ??= []).push(e), (t.f & b) !== 0 && (t.f & A) === 0 && (ft(
        /** @type {Derived} */
        t
      ), Ot(
        /** @type {Derived} */
        t
      ));
}
function Ct(e) {
  if (e.v === g) return !0;
  if (e.deps === null) return !1;
  for (const t of e.deps)
    if (K.has(t) || (t.f & b) !== 0 && Ct(
      /** @type {Derived} */
      t
    ))
      return !0;
  return !1;
}
function rr(e) {
  var t = P;
  try {
    return P = !0, e();
  } finally {
    P = t;
  }
}
function sr(e) {
  if (!(typeof e != "object" || !e || e instanceof EventTarget)) {
    if (ue in e)
      Ne(e);
    else if (!Array.isArray(e))
      for (let t in e) {
        const n = e[t];
        typeof n == "object" && n && ue in n && Ne(n);
      }
  }
}
function Ne(e, t = /* @__PURE__ */ new Set()) {
  if (typeof e == "object" && e !== null && // We don't want to traverse DOM elements
  !(e instanceof EventTarget) && !t.has(e)) {
    t.add(e), e instanceof Date && e.getTime();
    for (let r in e)
      try {
        Ne(e[r], t);
      } catch {
      }
    const n = Je(e);
    if (n !== Object.prototype && n !== Array.prototype && n !== Map.prototype && n !== Set.prototype && n !== Date.prototype) {
      const r = Ft(n);
      for (let s in r) {
        const l = r[s].get;
        if (l)
          try {
            l.call(e);
          } catch {
          }
      }
    }
  }
}
export {
  qt as $,
  Yt as A,
  Mn as B,
  Bt as C,
  ie as D,
  ye as E,
  z as F,
  D as G,
  Fe as H,
  An as I,
  On as J,
  on as K,
  B as L,
  jn as M,
  Pn as N,
  Gn as O,
  Rn as P,
  Yn as Q,
  Un as R,
  ue as S,
  Cn as T,
  Jn as U,
  zn as V,
  gt as W,
  Zn as X,
  _t as Y,
  Ge as Z,
  Oe as _,
  Hn as a,
  ln as a0,
  te as a1,
  H as a2,
  Ee as a3,
  W as a4,
  rn as a5,
  De as a6,
  be as a7,
  h as a8,
  _e as a9,
  Ln as aa,
  qn as ab,
  Nn as ac,
  Bn as ad,
  Xn as ae,
  Dn as af,
  p as b,
  Vn as c,
  X as d,
  Tn as e,
  er as f,
  pt as g,
  w as h,
  pn as i,
  Qn as j,
  Je as k,
  Ft as l,
  nr as m,
  S as n,
  $n as o,
  bn as p,
  Lt as q,
  tr as r,
  Kn as s,
  rr as t,
  Wn as u,
  Fn as v,
  le as w,
  sr as x,
  it as y,
  fe as z
};
