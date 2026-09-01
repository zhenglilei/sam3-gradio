import { i as un, g as $n, o as Yi, n as ut, u as he, s as Ji, r as zr, m as yt, a as H, b as l, t as fn, d as Qi, q as ei, c as ti, e as ft, f as pr, h as fr, j as Ki, T as $i, k as ea, l as cr, p as xt, v as cn, w as wt, x as ri, y as ni, z as ii, A as Wt, E as mr, B as Mt, C as ai, D as Le, F as En, G as ta, H as si, I as hn, J as ra, K as wn, L as na, M as ia, N as ze, O as oi, P as Nr, Q as aa, R as sa, S as oa, U as li, V as la, W as ua, X as ui, Y as dn, Z as Tn, _ as Sn, $ as fa, a0 as ca, a1 as ha, a2 as da, a3 as va, a4 as pa, a5 as ma, a6 as ga, a7 as vn, a8 as ba, a9 as fi, aa as gr, ab as _a, ac as ya, ad as xa, ae as Ea, af as wa, ag as Ta, ah as pn, ai as Sa, aj as Aa, ak as Ne, al as Xr, am as qr, an as Ha, ao as Ma, ap as Xt, aq as Oa, ar as Pa, as as Na, at as Ia, au as Ba, av as ci, aw as Ut, ax as Q, ay as La, az as An, aA as Ca, aB as xe, aC as br, aD as _r, aE as W, aF as Ie, aG as ne, aH as Ra, aI as ae, aJ as ge, aK as Be, aL as ka } from "./render-CPsC5Y6J.js";
function hi(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const Da = [];
function Ga(e, t = !1, r = !1) {
  return or(e, /* @__PURE__ */ new Map(), "", Da, null, r);
}
function or(e, t, r, n, i = null, a = !1) {
  if (typeof e == "object" && e !== null) {
    var o = t.get(e);
    if (o !== void 0) return o;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (un(e)) {
      var u = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, u), i !== null && t.set(i, u);
      for (var c = 0; c < e.length; c += 1) {
        var f = e[c];
        c in e && (u[c] = or(f, t, r, n, null, a));
      }
      return u;
    }
    if ($n(e) === Yi) {
      u = {}, t.set(e, u), i !== null && t.set(i, u);
      for (var d of Object.keys(e))
        u[d] = or(
          // @ts-expect-error
          e[d],
          t,
          r,
          n,
          null,
          a
        );
      return u;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !a)
      return or(
        /** @type {T & { toJSON(): any } } */
        e.toJSON(),
        t,
        r,
        n,
        // Associate the instance with the toJSON clone
        e
      );
  }
  if (e instanceof EventTarget)
    return (
      /** @type {Snapshot<T>} */
      e
    );
  try {
    return (
      /** @type {Snapshot<T>} */
      structuredClone(e)
    );
  } catch {
    return (
      /** @type {Snapshot<T>} */
      e
    );
  }
}
function mn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), ut;
  const n = he(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const St = [];
function Ua(e, t) {
  return {
    subscribe: Zt(e, t).subscribe
  };
}
function Zt(e, t = ut) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(u) {
    if (Ji(e, u) && (e = u, r)) {
      const c = !St.length;
      for (const f of n)
        f[1](), St.push(f, e);
      if (c) {
        for (let f = 0; f < St.length; f += 2)
          St[f][0](St[f + 1]);
        St.length = 0;
      }
    }
  }
  function a(u) {
    i(u(
      /** @type {T} */
      e
    ));
  }
  function o(u, c = ut) {
    const f = [u, c];
    return n.add(f), n.size === 1 && (r = t(i, a) || ut), u(
      /** @type {T} */
      e
    ), () => {
      n.delete(f), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function Lt(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return Ua(r, (o, u) => {
    let c = !1;
    const f = [];
    let d = 0, b = ut;
    const m = () => {
      if (d)
        return;
      b();
      const M = t(n ? f[0] : f, o, u);
      a ? o(M) : b = typeof M == "function" ? M : ut;
    }, T = i.map(
      (M, S) => mn(
        M,
        (y) => {
          f[S] = y, d &= ~(1 << S), c && m();
        },
        () => {
          d |= 1 << S;
        }
      )
    );
    return c = !0, m(), function() {
      zr(T), b(), c = !1;
    };
  });
}
function Fa(e) {
  let t;
  return mn(e, (r) => t = r)(), t;
}
let ir = !1, Wr = /* @__PURE__ */ Symbol("unmounted");
function Hn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: yt(void 0),
    unsubscribe: ut
  };
  if (n.store !== e && !(Wr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = ut;
    else {
      var i = !0;
      n.unsubscribe = mn(e, (a) => {
        i ? n.source.v = a : H(n.source, a);
      }), i = !1;
    }
  return e && Wr in r ? Fa(e) : l(n.source);
}
function ja() {
  const e = {};
  function t() {
    fn(() => {
      for (var r in e)
        e[r].unsubscribe();
      Qi(e, Wr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Va(e) {
  var t = ir;
  try {
    return ir = !1, [e(), ir];
  } finally {
    ir = t;
  }
}
function za(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, ei(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Xa = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function qa(e) {
  return (
    /** @type {string} */
    Xa?.createHTML(e) ?? e
  );
}
function di(e) {
  var t = ti("template");
  return t.innerHTML = qa(e.replaceAll("<!>", "<!---->")), t.content;
}
function Pt(e, t) {
  var r = (
    /** @type {Effect} */
    pr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function de(e, t) {
  var r = (t & $i) !== 0, n = (t & ea) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = di(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    fr(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ki ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var u = (
        /** @type {TemplateNode} */
        fr(o)
      ), c = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      Pt(u, c);
    } else
      Pt(o, o);
    return o;
  };
}
// @__NO_SIDE_EFFECTS__
function Wa(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var o = (
        /** @type {DocumentFragment} */
        di(i)
      ), u = (
        /** @type {Element} */
        fr(o)
      );
      a = /** @type {Element} */
      fr(u);
    }
    var c = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Pt(c, c), c;
  };
}
// @__NO_SIDE_EFFECTS__
function vi(e, t) {
  return /* @__PURE__ */ Wa(e, t, "svg");
}
function Ve(e = "") {
  {
    var t = ft(e + "");
    return Pt(t, t), t;
  }
}
function Ht() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = ft();
  return e.append(t, r), Pt(t, r), e;
}
function U(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class yr {
  /** @type {TemplateNode} */
  anchor;
  /** @type {Map<Batch, Key>} */
  #t = /* @__PURE__ */ new Map();
  /**
   * Map of keys to effects that are currently rendered in the DOM.
   * These effects are visible and actively part of the document tree.
   * Example:
   * ```
   * {#if condition}
   * 	foo
   * {:else}
   * 	bar
   * {/if}
   * ```
   * Can result in the entries `true->Effect` and `false->Effect`
   * @type {Map<Key, Effect>}
   */
  #r = /* @__PURE__ */ new Map();
  /**
   * Similar to #onscreen with respect to the keys, but contains branches that are not yet
   * in the DOM, because their insertion is deferred.
   * @type {Map<Key, Branch>}
   */
  #e = /* @__PURE__ */ new Map();
  /**
   * Keys of effects that are currently outroing
   * @type {Set<Key>}
   */
  #n = /* @__PURE__ */ new Set();
  /**
   * Whether to pause (i.e. outro) on change, or destroy immediately.
   * This is necessary for `<svelte:element>`
   */
  #i = !0;
  /**
   * @param {TemplateNode} anchor
   * @param {boolean} transition
   */
  constructor(t, r = !0) {
    this.anchor = t, this.#i = r;
  }
  /**
   * @param {Batch} batch
   */
  #a = (t) => {
    if (this.#t.has(t)) {
      var r = (
        /** @type {Key} */
        this.#t.get(t)
      ), n = this.#r.get(r);
      if (n)
        cr(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (cr(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const u = this.#e.get(o);
        u && (xt(u.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const u = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var f = document.createDocumentFragment();
            ni(o, f), f.append(ft()), this.#e.set(a, { effect: o, fragment: f });
          } else
            xt(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), cn(o, u, !1)) : u();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #s = (t) => {
    this.#t.delete(t);
    const r = Array.from(this.#t.values());
    for (const [n, i] of this.#e)
      r.includes(n) || (xt(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      ri
    ), i = ii();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = ft();
        a.append(o), this.#e.set(t, {
          effect: wt(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          wt(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [u, c] of this.#r)
        u === t ? n.unskip_effect(c) : n.skip_effect(c);
      for (const [u, c] of this.#e)
        u === t ? n.unskip_effect(c.effect) : n.skip_effect(c.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Za(e, t, ...r) {
  var n = new yr(e);
  Wt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, mr);
}
function Ya(e) {
  Mt === null && hi(), ai && Mt.l !== null ? Qa(Mt).m.push(e) : Le(() => {
    const t = he(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ja(e) {
  Mt === null && hi(), Ya(() => () => he(e));
}
function Qa(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function re(e, t, r = !1) {
  var n = new yr(e), i = r ? mr : 0;
  function a(o, u) {
    n.ensure(o, u);
  }
  Wt(() => {
    var o = !1;
    t((u, c = 0) => {
      o = !0, a(c, u);
    }), o || a(-1, null);
  }, i);
}
function Zr(e, t) {
  return t;
}
function Ka(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, u = 0; u < i; u++) {
    let b = t[u];
    cn(
      b,
      () => {
        if (a) {
          if (a.pending.delete(b), a.done.add(b), a.pending.size === 0) {
            var m = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Yr(e, hn(a.done)), m.delete(a), m.size === 0 && (e.outrogroups = null);
          }
        } else
          o -= 1;
      },
      !1
    );
  }
  if (o === 0) {
    var c = n.length === 0 && r !== null;
    if (c) {
      var f = (
        /** @type {Element} */
        r
      ), d = (
        /** @type {Element} */
        f.parentNode
      );
      sa(d), d.append(f), e.items.clear();
    }
    Yr(e, t, !c);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Yr(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const o of e.pending.values())
      for (const u of o)
        n.add(
          /** @type {EachItem} */
          e.items.get(u).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var a = t[i];
    if (n?.has(a)) {
      a.f |= ze;
      const o = document.createDocumentFragment();
      ni(a, o);
    } else
      xt(t[i], r);
  }
}
var Mn;
function Jr(e, t, r, n, i, a = null) {
  var o = e, u = /* @__PURE__ */ new Map(), c = (t & li) !== 0;
  if (c) {
    var f = (
      /** @type {Element} */
      e
    );
    o = f.appendChild(ft());
  }
  var d = null, b = si(() => {
    var g = r();
    return (
      /** @type {V[]} */
      un(g) ? g : g == null ? [] : hn(g)
    );
  }), m, T = /* @__PURE__ */ new Map(), M = !0;
  function S(g) {
    (p.effect.f & oi) === 0 && (p.pending.delete(g), p.fallback = d, $a(p, m, o, t, n), d !== null && (m.length === 0 ? (d.f & ze) === 0 ? cr(d) : (d.f ^= ze, Vt(d, null, o)) : cn(d, () => {
      d = null;
    })));
  }
  function y(g) {
    p.pending.delete(g);
  }
  var v = Wt(() => {
    m = /** @type {V[]} */
    l(b);
    for (var g = m.length, x = /* @__PURE__ */ new Set(), E = (
      /** @type {Batch} */
      ri
    ), P = ii(), N = 0; N < g; N += 1) {
      var k = m[N], D = n(k, N), C = M ? null : u.get(D);
      C ? (C.v && En(C.v, k), C.i && En(C.i, N), P && E.unskip_effect(C.e)) : (C = es(
        u,
        M ? o : Mn ??= ft(),
        k,
        D,
        N,
        i,
        t,
        r
      ), M || (C.e.f |= ze), u.set(D, C)), x.add(D);
    }
    if (g === 0 && a && !d && (M ? d = wt(() => a(o)) : (d = wt(() => a(Mn ??= ft())), d.f |= ze)), g > x.size && ta(), !M)
      if (T.set(E, x), P) {
        for (const [le, be] of u)
          x.has(le) || E.skip_effect(be.e);
        E.oncommit(S), E.ondiscard(y);
      } else
        S(E);
    l(b);
  }), p = { effect: v, items: u, pending: T, outrogroups: null, fallback: d };
  M = !1;
}
function Ft(e) {
  for (; e !== null && (e.f & aa) === 0; )
    e = e.next;
  return e;
}
function $a(e, t, r, n, i) {
  var a = (n & la) !== 0, o = t.length, u = e.items, c = Ft(e.effect.first), f, d = null, b, m = [], T = [], M, S, y, v;
  if (a)
    for (v = 0; v < o; v += 1)
      M = t[v], S = i(M, v), y = /** @type {EachItem} */
      u.get(S).e, (y.f & ze) === 0 && (y.nodes?.a?.measure(), (b ??= /* @__PURE__ */ new Set()).add(y));
  for (v = 0; v < o; v += 1) {
    if (M = t[v], S = i(M, v), y = /** @type {EachItem} */
    u.get(S).e, e.outrogroups !== null)
      for (const C of e.outrogroups)
        C.pending.delete(y), C.done.delete(y);
    if ((y.f & Nr) !== 0 && (cr(y), a && (y.nodes?.a?.unfix(), (b ??= /* @__PURE__ */ new Set()).delete(y))), (y.f & ze) !== 0)
      if (y.f ^= ze, y === c)
        Vt(y, null, r);
      else {
        var p = d ? d.next : c;
        y === e.effect.last && (e.effect.last = y.prev), y.prev && (y.prev.next = y.next), y.next && (y.next.prev = y.prev), ot(e, d, y), ot(e, y, p), Vt(y, p, r), d = y, m = [], T = [], c = Ft(d.next);
        continue;
      }
    if (y !== c) {
      if (f !== void 0 && f.has(y)) {
        if (m.length < T.length) {
          var g = T[0], x;
          d = g.prev;
          var E = m[0], P = m[m.length - 1];
          for (x = 0; x < m.length; x += 1)
            Vt(m[x], g, r);
          for (x = 0; x < T.length; x += 1)
            f.delete(T[x]);
          ot(e, E.prev, P.next), ot(e, d, E), ot(e, P, g), c = g, d = P, v -= 1, m = [], T = [];
        } else
          f.delete(y), Vt(y, c, r), ot(e, y.prev, y.next), ot(e, y, d === null ? e.effect.first : d.next), ot(e, d, y), d = y;
        continue;
      }
      for (m = [], T = []; c !== null && c !== y; )
        (f ??= /* @__PURE__ */ new Set()).add(c), T.push(c), c = Ft(c.next);
      if (c === null)
        continue;
    }
    (y.f & ze) === 0 && m.push(y), d = y, c = Ft(y.next);
  }
  if (e.outrogroups !== null) {
    for (const C of e.outrogroups)
      C.pending.size === 0 && (Yr(e, hn(C.done)), e.outrogroups?.delete(C));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (c !== null || f !== void 0) {
    var N = [];
    if (f !== void 0)
      for (y of f)
        (y.f & Nr) === 0 && N.push(y);
    for (; c !== null; )
      (c.f & Nr) === 0 && c !== e.fallback && N.push(c), c = Ft(c.next);
    var k = N.length;
    if (k > 0) {
      var D = (n & li) !== 0 && o === 0 ? r : null;
      if (a) {
        for (v = 0; v < k; v += 1)
          N[v].nodes?.a?.measure();
        for (v = 0; v < k; v += 1)
          N[v].nodes?.a?.fix();
      }
      Ka(e, N, D);
    }
  }
  a && ei(() => {
    if (b !== void 0)
      for (y of b)
        y.nodes?.a?.apply();
  });
}
function es(e, t, r, n, i, a, o, u) {
  var c = (o & na) !== 0 ? (o & ia) === 0 ? yt(r, !1, !1) : wn(r) : null, f = (o & ra) !== 0 ? wn(i) : null;
  return {
    v: c,
    i: f,
    e: wt(() => (a(t, c ?? r, f ?? i, u), () => {
      e.delete(n);
    }))
  };
}
function Vt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & ze) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        oa(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function ot(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Qr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function ts(e, t, r) {
  var n = new yr(e);
  Wt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, mr);
}
const rs = () => performance.now(), ke = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => rs(),
  tasks: /* @__PURE__ */ new Set()
};
function pi() {
  const e = ke.now();
  ke.tasks.forEach((t) => {
    t.c(e) || (ke.tasks.delete(t), t.f());
  }), ke.tasks.size !== 0 && ke.tick(pi);
}
function ns(e) {
  let t;
  return ke.tasks.size === 0 && ke.tick(pi), {
    promise: new Promise((r) => {
      ke.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      ke.tasks.delete(t);
    }
  };
}
function is(e, t, r, n, i, a) {
  var o = null, u = (
    /** @type {TemplateNode} */
    e
  ), c = new yr(u, !1);
  Wt(() => {
    const f = t() || null;
    var d = f === "svg" ? ua : void 0;
    if (f === null) {
      c.ensure(null, null);
      return;
    }
    return c.ensure(f, (b) => {
      if (f) {
        if (o = ti(f, d), Pt(o, o), n) {
          var m = null, T = o.appendChild(ft());
          n(o, T), m?.remove();
        }
        pr.nodes.end = o, b.before(o);
      }
    }), () => {
    };
  }, mr), fn(() => {
  });
}
function as(e, t) {
  var r = void 0, n;
  ui(() => {
    r !== (r = t()) && (n && (xt(n), n = null), r && (n = wt(() => {
      dn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function mi(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = mi(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function ss() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = mi(e)) && (n && (n += " "), n += t);
  return n;
}
function os(e) {
  return typeof e == "object" ? ss(e) : e ?? "";
}
const On = [...`\x20\t\n\r\f\u00A0\v\uFEFF`];
function ls(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var u = o + a;
          (o === 0 || On.includes(n[o - 1])) && (u === n.length || On.includes(n[u])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(u + 1) : o = u;
        }
  }
  return n === "" ? null : n;
}
function Pn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function Ir(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function us(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, u = !1, c = [];
      n && c.push(...Object.keys(n).map(Ir)), i && c.push(...Object.keys(i).map(Ir));
      var f = 0, d = -1;
      const S = e.length;
      for (var b = 0; b < S; b++) {
        var m = e[b];
        if (u ? m === "/" && e[b - 1] === "*" && (u = !1) : a ? a === m && (a = !1) : m === "/" && e[b + 1] === "*" ? u = !0 : m === '"' || m === "'" ? a = m : m === "(" ? o++ : m === ")" && o--, !u && a === !1 && o === 0) {
          if (m === ":" && d === -1)
            d = b;
          else if (m === ";" || b === S - 1) {
            if (d !== -1) {
              var T = Ir(e.substring(f, d).trim());
              if (!c.includes(T)) {
                m !== ";" && b++;
                var M = e.substring(f, b).trim();
                r += " " + M + ";";
              }
            }
            f = b + 1, d = -1;
          }
        }
      }
    }
    return n && (r += Pn(n)), i && (r += Pn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Et(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[Tn]
  );
  if (o !== r || o === void 0) {
    var u = ls(r, n, a);
    u == null ? e.removeAttribute("class") : t ? e.className = u : e.setAttribute("class", u), e[Tn] = r;
  } else if (a && i !== a)
    for (var c in a) {
      var f = !!a[c];
      (i == null || f !== !!i[c]) && e.classList.toggle(c, f);
    }
  return a;
}
function Br(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function De(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[Sn]
  );
  if (i !== t) {
    var a = us(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[Sn] = t;
  } else n && (Array.isArray(n) ? (Br(e, r?.[0], n[0]), Br(e, r?.[1], n[1], "important")) : Br(e, r, n));
  return n;
}
function hr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!un(t))
      return fa();
    for (var n of e.options)
      n.selected = t.includes(Nn(n));
    return;
  }
  for (n of e.options) {
    var i = Nn(n);
    if (ca(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function gi(e) {
  var t = new MutationObserver(() => {
    hr(e, e.__value);
  });
  t.observe(e, {
    // Listen to option element changes
    childList: !0,
    subtree: !0,
    // because of <optgroup>
    // Listen to option element value attribute changes
    // (doesn't get notified of select value changes,
    // because that property is not reflected as an attribute)
    attributes: !0,
    attributeFilter: ["value"]
  }), fn(() => {
    t.disconnect();
  });
}
function Nn(e) {
  return "__value" in e ? e.__value : e.value;
}
const zt = /* @__PURE__ */ Symbol("class"), At = /* @__PURE__ */ Symbol("style"), bi = /* @__PURE__ */ Symbol("is custom element"), _i = /* @__PURE__ */ Symbol("is html"), fs = vn ? "input" : "INPUT", cs = vn ? "option" : "OPTION", hs = vn ? "select" : "SELECT";
function ds(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function Ot(e, t, r, n) {
  var i = yi(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[ha] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && xi(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function vs(e, t, r, n, i = !1, a = !1) {
  var o = yi(e), u = o[bi], c = !o[_i], f = t || {}, d = e.nodeName === cs;
  for (var b in t)
    b in r || (r[b] = null);
  r.class ? r.class = os(r.class) : r.class = null, r[At] && (r.style ??= null);
  var m = xi(e);
  if (e.nodeName === fs && "type" in r && ("value" in r || "__value" in r)) {
    var T = r.type;
    (T !== f.type || T === void 0 && e.hasAttribute("type")) && (f.type = T, Ot(e, "type", T));
  }
  for (const x in r) {
    let E = r[x];
    if (d && x === "value" && E == null) {
      e.value = e.__value = "", f[x] = E;
      continue;
    }
    if (x === "class") {
      var M = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Et(e, M, E, n, t?.[zt], r[zt]), f[x] = E, f[zt] = r[zt];
      continue;
    }
    if (x === "style") {
      De(e, E, t?.[At], r[At]), f[x] = E, f[At] = r[At];
      continue;
    }
    var S = f[x];
    if (!(E === S && !(E === void 0 && e.hasAttribute(x)))) {
      f[x] = E;
      var y = x[0] + x[1];
      if (y !== "$$")
        if (y === "on") {
          const P = {}, N = "$$" + x;
          let k = x.slice(2);
          var v = Ea(k);
          if (ba(k) && (k = k.slice(0, -7), P.capture = !0), !v && S) {
            if (E != null) continue;
            e.removeEventListener(k, f[N], P), f[N] = null;
          }
          if (v)
            fi(k, e, E), gr([k]);
          else if (E != null) {
            let D = function(C) {
              f[x].call(this, C);
            };
            f[N] = _a(k, e, D, P);
          }
        } else if (x === "style")
          Ot(e, x, E);
        else if (x === "autofocus")
          za(
            /** @type {HTMLElement} */
            e,
            !!E
          );
        else if (!u && (x === "__value" || x === "value" && E != null))
          e.value = e.__value = E;
        else if (x === "selected" && d)
          ds(
            /** @type {HTMLOptionElement} */
            e,
            E
          );
        else {
          var p = x;
          c || (p = ya(p));
          var g = p === "defaultValue" || p === "defaultChecked";
          if (E == null && !u && !g)
            if (o[x] = null, p === "value" || p === "checked") {
              let P = (
                /** @type {HTMLInputElement} */
                e
              );
              const N = t === void 0;
              if (p === "value") {
                let k = P.defaultValue;
                P.removeAttribute(p), P.defaultValue = k, P.value = P.__value = N ? k : null;
              } else {
                let k = P.defaultChecked;
                P.removeAttribute(p), P.defaultChecked = k, P.checked = N ? k : !1;
              }
            } else
              e.removeAttribute(x);
          else g || m.includes(p) && (u || typeof E != "string") ? (e[p] = E, p in o && (o[p] = xa)) : typeof E != "function" && Ot(e, p, E);
        }
    }
  }
  return f;
}
function ps(e, t, r = [], n = [], i = [], a, o = !1, u = !1) {
  ma(i, r, n, (c) => {
    var f = void 0, d = {}, b = e.nodeName === hs, m = !1;
    if (ui(() => {
      var M = t(...c.map(l)), S = vs(
        e,
        f,
        M,
        a,
        o,
        u
      );
      m && b && "value" in M && hr(
        /** @type {HTMLSelectElement} */
        e,
        M.value
      );
      for (let v of Object.getOwnPropertySymbols(d))
        M[v] || xt(d[v]);
      for (let v of Object.getOwnPropertySymbols(M)) {
        var y = M[v];
        v.description === ga && (!f || y !== f[v]) && (d[v] && xt(d[v]), d[v] = wt(() => as(e, () => y))), S[v] = y;
      }
      f = S;
    }), b) {
      var T = (
        /** @type {HTMLSelectElement} */
        e
      );
      dn(() => {
        hr(
          T,
          /** @type {Record<string | symbol, any>} */
          f.value,
          !0
        ), gi(T);
      });
    }
    m = !0;
  });
}
function yi(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[da] ??= {
      [bi]: e.nodeName.includes("-"),
      [_i]: e.namespaceURI === va
    }
  );
}
var In = /* @__PURE__ */ new Map();
function xi(e) {
  var t = e.getAttribute("is") || e.nodeName, r = In.get(t);
  if (r) return r;
  In.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = pa(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = $n(i);
  }
  return r;
}
function Lr(e, t) {
  return e === t || e?.[pn] === t;
}
function gn(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Mt.r
  ), a = (
    /** @type {Effect} */
    pr
  );
  return dn(() => {
    var o, u;
    return wa(() => {
      o = u, u = [], he(() => {
        Lr(r(...u), e) || (t(e, ...u), o && Lr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let c = a;
      for (; c !== i && c.parent !== null && c.parent.f & Ta; )
        c = c.parent;
      const f = () => {
        u && Lr(r(...u), e) && t(null, ...u);
      }, d = c.teardown;
      c.teardown = () => {
        f(), d?.();
      };
    };
  }), e;
}
function ms(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Mt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Ne(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Xr(() => {
      let u = !1;
      const c = t.s;
      for (const f in c)
        c[f] !== a[f] && (a[f] = c[f], u = !0);
      return u && i++, i;
    });
    n = () => l(o);
  }
  r.b.length && Sa(() => {
    Bn(t, n), zr(r.b);
  }), Le(() => {
    const i = he(() => r.m.map(Aa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Le(() => {
    Bn(t, n), zr(r.a);
  });
}
function Bn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) l(r);
  t();
}
const gs = {
  get(e, t) {
    if (!e.exclude.has(t))
      return e.props[t];
  },
  set(e, t) {
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    if (!e.exclude.has(t) && t in e.props)
      return {
        enumerable: !0,
        configurable: !0,
        value: e.props[t]
      };
  },
  has(e, t) {
    return e.exclude.has(t) ? !1 : t in e.props;
  },
  ownKeys(e) {
    return Reflect.ownKeys(e.props).filter((t) => !e.exclude.has(t));
  }
};
// @__NO_SIDE_EFFECTS__
function bs(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    gs
  );
}
const _s = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Ut(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Ut(i) && (i = i());
      const a = qr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Ut(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = qr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === pn || t === ci) return !1;
    for (let r of e.props)
      if (Ut(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Ut(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function ys(...e) {
  return new Proxy({ props: e }, _s);
}
function L(e, t, r, n) {
  var i = !ai || (r & Pa) !== 0, a = (r & Oa) !== 0, o = (r & Ia) !== 0, u = (
    /** @type {V} */
    n
  ), c = !0, f = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), d = () => o && i ? (f ??= Xr(
    /** @type {() => V} */
    n
  ), l(f)) : (c && (c = !1, u = o ? he(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), u);
  let b;
  if (a) {
    var m = pn in e || ci in e;
    b = qr(e, t)?.set ?? (m && t in e ? (x) => e[t] = x : void 0);
  }
  var T, M = !1;
  a ? [T, M] = Va(() => (
    /** @type {V} */
    e[t]
  )) : T = /** @type {V} */
  e[t], T === void 0 && n !== void 0 && (T = d(), b && (i && Ha(), b(T)));
  var S;
  if (i ? S = () => {
    var x = (
      /** @type {V} */
      e[t]
    );
    return x === void 0 ? d() : (c = !0, x);
  } : S = () => {
    var x = (
      /** @type {V} */
      e[t]
    );
    return x !== void 0 && (u = /** @type {V} */
    void 0), x === void 0 ? u : x;
  }, i && (r & Ma) === 0)
    return S;
  if (b) {
    var y = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(x, E) {
        return arguments.length > 0 ? ((!i || !E || y || M) && b(E ? S() : x), x) : S();
      })
    );
  }
  var v = !1, p = ((r & Na) !== 0 ? Xr : si)(() => (v = !1, S()));
  a && l(p);
  var g = (
    /** @type {Effect} */
    pr
  );
  return (
    /** @type {() => V} */
    (function(x, E) {
      if (arguments.length > 0) {
        const P = E ? l(p) : i && a ? Xt(x) : x;
        return H(p, P), v = !0, u !== void 0 && (u = P), x;
      }
      return Ba && v || (g.f & oi) !== 0 ? p.v : l(p);
    })
  );
}
const xs = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], Ln = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
xs.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: Ln[t][r],
    secondary: Ln[t][n]
  }
}), {});
function Es(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Cr, Cn;
function ws() {
  if (Cn) return Cr;
  Cn = 1;
  var e = function(p) {
    return t(p) && !r(p);
  };
  function t(v) {
    return !!v && typeof v == "object";
  }
  function r(v) {
    var p = Object.prototype.toString.call(v);
    return p === "[object RegExp]" || p === "[object Date]" || a(v);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(v) {
    return v.$$typeof === i;
  }
  function o(v) {
    return Array.isArray(v) ? [] : {};
  }
  function u(v, p) {
    return p.clone !== !1 && p.isMergeableObject(v) ? S(o(v), v, p) : v;
  }
  function c(v, p, g) {
    return v.concat(p).map(function(x) {
      return u(x, g);
    });
  }
  function f(v, p) {
    if (!p.customMerge)
      return S;
    var g = p.customMerge(v);
    return typeof g == "function" ? g : S;
  }
  function d(v) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(v).filter(function(p) {
      return Object.propertyIsEnumerable.call(v, p);
    }) : [];
  }
  function b(v) {
    return Object.keys(v).concat(d(v));
  }
  function m(v, p) {
    try {
      return p in v;
    } catch {
      return !1;
    }
  }
  function T(v, p) {
    return m(v, p) && !(Object.hasOwnProperty.call(v, p) && Object.propertyIsEnumerable.call(v, p));
  }
  function M(v, p, g) {
    var x = {};
    return g.isMergeableObject(v) && b(v).forEach(function(E) {
      x[E] = u(v[E], g);
    }), b(p).forEach(function(E) {
      T(v, E) || (m(v, E) && g.isMergeableObject(p[E]) ? x[E] = f(E, g)(v[E], p[E], g) : x[E] = u(p[E], g));
    }), x;
  }
  function S(v, p, g) {
    g = g || {}, g.arrayMerge = g.arrayMerge || c, g.isMergeableObject = g.isMergeableObject || e, g.cloneUnlessOtherwiseSpecified = u;
    var x = Array.isArray(p), E = Array.isArray(v), P = x === E;
    return P ? x ? g.arrayMerge(v, p, g) : M(v, p, g) : u(p, g);
  }
  S.all = function(p, g) {
    if (!Array.isArray(p))
      throw new Error("first argument should be an array");
    return p.reduce(function(x, E) {
      return S(x, E, g);
    }, {});
  };
  var y = S;
  return Cr = y, Cr;
}
var Ts = ws();
const Ss = /* @__PURE__ */ Es(Ts);
var Kr = function(e, t) {
  return Kr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Kr(e, t);
};
function xr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Kr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var z = function() {
  return z = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, z.apply(this, arguments);
};
function As(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function Rr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function kr(e, t) {
  var r = t && t.cache ? t.cache : Bs, n = t && t.serializer ? t.serializer : Ns, i = t && t.strategy ? t.strategy : Os;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function Hs(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Ms(e, t, r, n) {
  var i = Hs(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function Ei(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function wi(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function Os(e, t) {
  var r = e.length === 1 ? Ms : Ei;
  return wi(e, this, r, t.cache.create(), t.serializer);
}
function Ps(e, t) {
  return wi(e, this, Ei, t.cache.create(), t.serializer);
}
var Ns = function() {
  return JSON.stringify(arguments);
}, Is = (
  /** @class */
  (function() {
    function e() {
      this.cache = /* @__PURE__ */ Object.create(null);
    }
    return e.prototype.get = function(t) {
      return this.cache[t];
    }, e.prototype.set = function(t, r) {
      this.cache[t] = r;
    }, e;
  })()
), Bs = {
  create: function() {
    return new Is();
  }
}, Dr = {
  variadic: Ps
}, F;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(F || (F = {}));
var $;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})($ || ($ = {}));
var Nt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(Nt || (Nt = {}));
function Rn(e) {
  return e.type === $.literal;
}
function Ls(e) {
  return e.type === $.argument;
}
function Ti(e) {
  return e.type === $.number;
}
function Si(e) {
  return e.type === $.date;
}
function Ai(e) {
  return e.type === $.time;
}
function Hi(e) {
  return e.type === $.select;
}
function Mi(e) {
  return e.type === $.plural;
}
function Cs(e) {
  return e.type === $.pound;
}
function Oi(e) {
  return e.type === $.tag;
}
function Pi(e) {
  return !!(e && typeof e == "object" && e.type === Nt.number);
}
function $r(e) {
  return !!(e && typeof e == "object" && e.type === Nt.dateTime);
}
var Ni = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, Rs = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function ks(e) {
  var t = {};
  return e.replace(Rs, function(r) {
    var n = r.length;
    switch (r[0]) {
      // Era
      case "G":
        t.era = n === 4 ? "long" : n === 5 ? "narrow" : "short";
        break;
      // Year
      case "y":
        t.year = n === 2 ? "2-digit" : "numeric";
        break;
      case "Y":
      case "u":
      case "U":
      case "r":
        throw new RangeError("`Y/u/U/r` (year) patterns are not supported, use `y` instead");
      // Quarter
      case "q":
      case "Q":
        throw new RangeError("`q/Q` (quarter) patterns are not supported");
      // Month
      case "M":
      case "L":
        t.month = ["numeric", "2-digit", "short", "long", "narrow"][n - 1];
        break;
      // Week
      case "w":
      case "W":
        throw new RangeError("`w/W` (week) patterns are not supported");
      case "d":
        t.day = ["numeric", "2-digit"][n - 1];
        break;
      case "D":
      case "F":
      case "g":
        throw new RangeError("`D/F/g` (day) patterns are not supported, use `d` instead");
      // Weekday
      case "E":
        t.weekday = n === 4 ? "long" : n === 5 ? "narrow" : "short";
        break;
      case "e":
        if (n < 4)
          throw new RangeError("`e..eee` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][n - 4];
        break;
      case "c":
        if (n < 4)
          throw new RangeError("`c..ccc` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][n - 4];
        break;
      // Period
      case "a":
        t.hour12 = !0;
        break;
      case "b":
      // am, pm, noon, midnight
      case "B":
        throw new RangeError("`b/B` (period) patterns are not supported, use `a` instead");
      // Hour
      case "h":
        t.hourCycle = "h12", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "H":
        t.hourCycle = "h23", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "K":
        t.hourCycle = "h11", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "k":
        t.hourCycle = "h24", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "j":
      case "J":
      case "C":
        throw new RangeError("`j/J/C` (hour) patterns are not supported, use `h/H/K/k` instead");
      // Minute
      case "m":
        t.minute = ["numeric", "2-digit"][n - 1];
        break;
      // Second
      case "s":
        t.second = ["numeric", "2-digit"][n - 1];
        break;
      case "S":
      case "A":
        throw new RangeError("`S/A` (second) patterns are not supported, use `s` instead");
      // Zone
      case "z":
        t.timeZoneName = n < 4 ? "short" : "long";
        break;
      case "Z":
      // 1..3, 4, 5: The ISO8601 varios formats
      case "O":
      // 1, 4: milliseconds in day short, long
      case "v":
      // 1, 4: generic non-location format
      case "V":
      // 1, 2, 3, 4: time zone ID or city
      case "X":
      // 1, 2, 3, 4: The ISO8601 varios formats
      case "x":
        throw new RangeError("`Z/O/v/V/X/x` (timeZone) patterns are not supported, use `z` instead");
    }
    return "";
  }), t;
}
var Ds = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Gs(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Ds).filter(function(m) {
    return m.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var u = o[0], c = o.slice(1), f = 0, d = c; f < d.length; f++) {
      var b = d[f];
      if (b.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: u, options: c });
  }
  return r;
}
function Us(e) {
  return e.replace(/^(.*?)-/, "");
}
var kn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, Ii = /^(@+)?(\+|#+)?[rs]?$/g, Fs = /(\*)(0+)|(#+)(0+)|(0+)/g, Bi = /^(0+)$/;
function Dn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(Ii, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function Li(e) {
  switch (e) {
    case "sign-auto":
      return {
        signDisplay: "auto"
      };
    case "sign-accounting":
    case "()":
      return {
        currencySign: "accounting"
      };
    case "sign-always":
    case "+!":
      return {
        signDisplay: "always"
      };
    case "sign-accounting-always":
    case "()!":
      return {
        signDisplay: "always",
        currencySign: "accounting"
      };
    case "sign-except-zero":
    case "+?":
      return {
        signDisplay: "exceptZero"
      };
    case "sign-accounting-except-zero":
    case "()?":
      return {
        signDisplay: "exceptZero",
        currencySign: "accounting"
      };
    case "sign-never":
    case "+_":
      return {
        signDisplay: "never"
      };
  }
}
function js(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !Bi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Gn(e) {
  var t = {}, r = Li(e);
  return r || t;
}
function Vs(e) {
  for (var t = {}, r = 0, n = e; r < n.length; r++) {
    var i = n[r];
    switch (i.stem) {
      case "percent":
      case "%":
        t.style = "percent";
        continue;
      case "%x100":
        t.style = "percent", t.scale = 100;
        continue;
      case "currency":
        t.style = "currency", t.currency = i.options[0];
        continue;
      case "group-off":
      case ",_":
        t.useGrouping = !1;
        continue;
      case "precision-integer":
      case ".":
        t.maximumFractionDigits = 0;
        continue;
      case "measure-unit":
      case "unit":
        t.style = "unit", t.unit = Us(i.options[0]);
        continue;
      case "compact-short":
      case "K":
        t.notation = "compact", t.compactDisplay = "short";
        continue;
      case "compact-long":
      case "KK":
        t.notation = "compact", t.compactDisplay = "long";
        continue;
      case "scientific":
        t = z(z(z({}, t), { notation: "scientific" }), i.options.reduce(function(c, f) {
          return z(z({}, c), Gn(f));
        }, {}));
        continue;
      case "engineering":
        t = z(z(z({}, t), { notation: "engineering" }), i.options.reduce(function(c, f) {
          return z(z({}, c), Gn(f));
        }, {}));
        continue;
      case "notation-simple":
        t.notation = "standard";
        continue;
      // https://github.com/unicode-org/icu/blob/master/icu4c/source/i18n/unicode/unumberformatter.h
      case "unit-width-narrow":
        t.currencyDisplay = "narrowSymbol", t.unitDisplay = "narrow";
        continue;
      case "unit-width-short":
        t.currencyDisplay = "code", t.unitDisplay = "short";
        continue;
      case "unit-width-full-name":
        t.currencyDisplay = "name", t.unitDisplay = "long";
        continue;
      case "unit-width-iso-code":
        t.currencyDisplay = "symbol";
        continue;
      case "scale":
        t.scale = parseFloat(i.options[0]);
        continue;
      case "rounding-mode-floor":
        t.roundingMode = "floor";
        continue;
      case "rounding-mode-ceiling":
        t.roundingMode = "ceil";
        continue;
      case "rounding-mode-down":
        t.roundingMode = "trunc";
        continue;
      case "rounding-mode-up":
        t.roundingMode = "expand";
        continue;
      case "rounding-mode-half-even":
        t.roundingMode = "halfEven";
        continue;
      case "rounding-mode-half-down":
        t.roundingMode = "halfTrunc";
        continue;
      case "rounding-mode-half-up":
        t.roundingMode = "halfExpand";
        continue;
      // https://unicode-org.github.io/icu/userguide/format_parse/numbers/skeletons.html#integer-width
      case "integer-width":
        if (i.options.length > 1)
          throw new RangeError("integer-width stems only accept a single optional option");
        i.options[0].replace(Fs, function(c, f, d, b, m, T) {
          if (f)
            t.minimumIntegerDigits = d.length;
          else {
            if (b && m)
              throw new Error("We currently do not support maximum integer digits");
            if (T)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (Bi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (kn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(kn, function(c, f, d, b, m, T) {
        return d === "*" ? t.minimumFractionDigits = f.length : b && b[0] === "#" ? t.maximumFractionDigits = b.length : m && T ? (t.minimumFractionDigits = m.length, t.maximumFractionDigits = m.length + T.length) : (t.minimumFractionDigits = f.length, t.maximumFractionDigits = f.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = z(z({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = z(z({}, t), Dn(a)));
      continue;
    }
    if (Ii.test(i.stem)) {
      t = z(z({}, t), Dn(i.stem));
      continue;
    }
    var o = Li(i.stem);
    o && (t = z(z({}, t), o));
    var u = js(i.stem);
    u && (t = z(z({}, t), u));
  }
  return t;
}
var ar = {
  "001": [
    "H",
    "h"
  ],
  419: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  AC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  AD: [
    "H",
    "hB"
  ],
  AE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  AF: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  AG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  AI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  AL: [
    "h",
    "H",
    "hB"
  ],
  AM: [
    "H",
    "hB"
  ],
  AO: [
    "H",
    "hB"
  ],
  AR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  AS: [
    "h",
    "H"
  ],
  AT: [
    "H",
    "hB"
  ],
  AU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  AW: [
    "H",
    "hB"
  ],
  AX: [
    "H"
  ],
  AZ: [
    "H",
    "hB",
    "h"
  ],
  BA: [
    "H",
    "hB",
    "h"
  ],
  BB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BD: [
    "h",
    "hB",
    "H"
  ],
  BE: [
    "H",
    "hB"
  ],
  BF: [
    "H",
    "hB"
  ],
  BG: [
    "H",
    "hB",
    "h"
  ],
  BH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  BI: [
    "H",
    "h"
  ],
  BJ: [
    "H",
    "hB"
  ],
  BL: [
    "H",
    "hB"
  ],
  BM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BN: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  BO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  BQ: [
    "H"
  ],
  BR: [
    "H",
    "hB"
  ],
  BS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BT: [
    "h",
    "H"
  ],
  BW: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  BY: [
    "H",
    "h"
  ],
  BZ: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CA: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  CC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CD: [
    "hB",
    "H"
  ],
  CF: [
    "H",
    "h",
    "hB"
  ],
  CG: [
    "H",
    "hB"
  ],
  CH: [
    "H",
    "hB",
    "h"
  ],
  CI: [
    "H",
    "hB"
  ],
  CK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CL: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CM: [
    "H",
    "h",
    "hB"
  ],
  CN: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  CO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CP: [
    "H"
  ],
  CR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CU: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CV: [
    "H",
    "hB"
  ],
  CW: [
    "H",
    "hB"
  ],
  CX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CY: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  CZ: [
    "H"
  ],
  DE: [
    "H",
    "hB"
  ],
  DG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  DJ: [
    "h",
    "H"
  ],
  DK: [
    "H"
  ],
  DM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  DO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  DZ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  EC: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  EE: [
    "H",
    "hB"
  ],
  EG: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  ER: [
    "h",
    "H"
  ],
  ES: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  ET: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  FI: [
    "H"
  ],
  FJ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  FM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FO: [
    "H",
    "h"
  ],
  FR: [
    "H",
    "hB"
  ],
  GA: [
    "H",
    "hB"
  ],
  GB: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GD: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GE: [
    "H",
    "hB",
    "h"
  ],
  GF: [
    "H",
    "hB"
  ],
  GG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GH: [
    "h",
    "H"
  ],
  GI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GL: [
    "H",
    "h"
  ],
  GM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GN: [
    "H",
    "hB"
  ],
  GP: [
    "H",
    "hB"
  ],
  GQ: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  GR: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  GT: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  GU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GW: [
    "H",
    "hB"
  ],
  GY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  HK: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  HN: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  HR: [
    "H",
    "hB"
  ],
  HU: [
    "H",
    "h"
  ],
  IC: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  ID: [
    "H"
  ],
  IE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IL: [
    "H",
    "hB"
  ],
  IM: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IN: [
    "h",
    "H"
  ],
  IO: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IQ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  IR: [
    "hB",
    "H"
  ],
  IS: [
    "H"
  ],
  IT: [
    "H",
    "hB"
  ],
  JE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  JM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  JO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  JP: [
    "H",
    "K",
    "h"
  ],
  KE: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  KG: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KH: [
    "hB",
    "h",
    "H",
    "hb"
  ],
  KI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KM: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KN: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KP: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KW: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  KY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KZ: [
    "H",
    "hB"
  ],
  LA: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  LB: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  LC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LI: [
    "H",
    "hB",
    "h"
  ],
  LK: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  LR: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LS: [
    "h",
    "H"
  ],
  LT: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  LU: [
    "H",
    "h",
    "hB"
  ],
  LV: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  LY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  MC: [
    "H",
    "hB"
  ],
  MD: [
    "H",
    "hB"
  ],
  ME: [
    "H",
    "hB",
    "h"
  ],
  MF: [
    "H",
    "hB"
  ],
  MG: [
    "H",
    "h"
  ],
  MH: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  ML: [
    "H"
  ],
  MM: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  MN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MP: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MQ: [
    "H",
    "hB"
  ],
  MR: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MS: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MT: [
    "H",
    "h"
  ],
  MU: [
    "H",
    "h"
  ],
  MV: [
    "H",
    "h"
  ],
  MW: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MX: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  MY: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  MZ: [
    "H",
    "hB"
  ],
  NA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  NC: [
    "H",
    "hB"
  ],
  NE: [
    "H"
  ],
  NF: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NI: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  NL: [
    "H",
    "hB"
  ],
  NO: [
    "H",
    "h"
  ],
  NP: [
    "H",
    "h",
    "hB"
  ],
  NR: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NU: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  OM: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PE: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PF: [
    "H",
    "h",
    "hB"
  ],
  PG: [
    "h",
    "H"
  ],
  PH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PK: [
    "h",
    "hB",
    "H"
  ],
  PL: [
    "H",
    "h"
  ],
  PM: [
    "H",
    "hB"
  ],
  PN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  PR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PS: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PT: [
    "H",
    "hB"
  ],
  PW: [
    "h",
    "H"
  ],
  PY: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  QA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  RE: [
    "H",
    "hB"
  ],
  RO: [
    "H",
    "hB"
  ],
  RS: [
    "H",
    "hB",
    "h"
  ],
  RU: [
    "H"
  ],
  RW: [
    "H",
    "h"
  ],
  SA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SC: [
    "H",
    "h",
    "hB"
  ],
  SD: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SE: [
    "H"
  ],
  SG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SH: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SI: [
    "H",
    "hB"
  ],
  SJ: [
    "H"
  ],
  SK: [
    "H"
  ],
  SL: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SM: [
    "H",
    "h",
    "hB"
  ],
  SN: [
    "H",
    "h",
    "hB"
  ],
  SO: [
    "h",
    "H"
  ],
  SR: [
    "H",
    "hB"
  ],
  SS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  ST: [
    "H",
    "hB"
  ],
  SV: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  SX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  TC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TD: [
    "h",
    "H",
    "hB"
  ],
  TF: [
    "H",
    "h",
    "hB"
  ],
  TG: [
    "H",
    "hB"
  ],
  TH: [
    "H",
    "h"
  ],
  TJ: [
    "H",
    "h"
  ],
  TL: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  TM: [
    "H",
    "h"
  ],
  TN: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  TO: [
    "h",
    "H"
  ],
  TR: [
    "H",
    "hB"
  ],
  TT: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TW: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  TZ: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  UA: [
    "H",
    "hB",
    "h"
  ],
  UG: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  UM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  US: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  UY: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  UZ: [
    "H",
    "hB",
    "h"
  ],
  VA: [
    "H",
    "h",
    "hB"
  ],
  VC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VE: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  VG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VN: [
    "H",
    "h"
  ],
  VU: [
    "h",
    "H"
  ],
  WF: [
    "H",
    "hB"
  ],
  WS: [
    "h",
    "H"
  ],
  XK: [
    "H",
    "hB",
    "h"
  ],
  YE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  YT: [
    "H",
    "hB"
  ],
  ZA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  ZM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  ZW: [
    "H",
    "h"
  ],
  "af-ZA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "ar-001": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "ca-ES": [
    "H",
    "h",
    "hB"
  ],
  "en-001": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-HK": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-IL": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "en-MY": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "es-BR": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-ES": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-GQ": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "fr-CA": [
    "H",
    "h",
    "hB"
  ],
  "gl-ES": [
    "H",
    "h",
    "hB"
  ],
  "gu-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "hi-IN": [
    "hB",
    "h",
    "H"
  ],
  "it-CH": [
    "H",
    "h",
    "hB"
  ],
  "it-IT": [
    "H",
    "h",
    "hB"
  ],
  "kn-IN": [
    "hB",
    "h",
    "H"
  ],
  "ml-IN": [
    "hB",
    "h",
    "H"
  ],
  "mr-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "pa-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "ta-IN": [
    "hB",
    "h",
    "hb",
    "H"
  ],
  "te-IN": [
    "hB",
    "h",
    "H"
  ],
  "zu-ZA": [
    "H",
    "hB",
    "hb",
    "h"
  ]
};
function zs(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), u = a < 2 ? 1 : 3 + (a >> 1), c = "a", f = Xs(t);
      for ((f == "H" || f == "k") && (u = 0); u-- > 0; )
        r += c;
      for (; o-- > 0; )
        r = f + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Xs(e) {
  var t = e.hourCycle;
  if (t === void 0 && // @ts-ignore hourCycle(s) is not identified yet
  e.hourCycles && // @ts-ignore
  e.hourCycles.length && (t = e.hourCycles[0]), t)
    switch (t) {
      case "h24":
        return "k";
      case "h23":
        return "H";
      case "h12":
        return "h";
      case "h11":
        return "K";
      default:
        throw new Error("Invalid hourCycle");
    }
  var r = e.language, n;
  r !== "root" && (n = e.maximize().region);
  var i = ar[n || ""] || ar[r || ""] || ar["".concat(r, "-001")] || ar["001"];
  return i[0];
}
var Gr, qs = new RegExp("^".concat(Ni.source, "*")), Ws = new RegExp("".concat(Ni.source, "*$"));
function j(e, t) {
  return { start: e, end: t };
}
var Zs = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ys = !!String.fromCodePoint, Js = !!Object.fromEntries, Qs = !!String.prototype.codePointAt, Ks = !!String.prototype.trimStart, $s = !!String.prototype.trimEnd, eo = !!Number.isSafeInteger, to = eo ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, en = !0;
try {
  var ro = Ri("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  en = ((Gr = ro.exec("a")) === null || Gr === void 0 ? void 0 : Gr[0]) === "a";
} catch {
  en = !1;
}
var Un = Zs ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), tn = Ys ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", i = t.length, a = 0, o; i > a; ) {
      if (o = t[a++], o > 1114111)
        throw RangeError(o + " is not a valid code point");
      n += o < 65536 ? String.fromCharCode(o) : String.fromCharCode(((o -= 65536) >> 10) + 55296, o % 1024 + 56320);
    }
    return n;
  }
), Fn = (
  // native
  Js ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], u = a[1];
        r[o] = u;
      }
      return r;
    }
  )
), Ci = Qs ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var i = t.charCodeAt(r), a;
      return i < 55296 || i > 56319 || r + 1 === n || (a = t.charCodeAt(r + 1)) < 56320 || a > 57343 ? i : (i - 55296 << 10) + (a - 56320) + 65536;
    }
  }
), no = Ks ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(qs, "");
  }
), io = $s ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ws, "");
  }
);
function Ri(e, t) {
  return new RegExp(e, t);
}
var rn;
if (en) {
  var jn = Ri("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  rn = function(t, r) {
    var n;
    jn.lastIndex = r;
    var i = jn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  rn = function(t, r) {
    for (var n = []; ; ) {
      var i = Ci(t, r);
      if (i === void 0 || ki(i) || lo(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return tn.apply(void 0, n);
  };
var ao = (
  /** @class */
  (function() {
    function e(t, r) {
      r === void 0 && (r = {}), this.message = t, this.position = { offset: 0, line: 1, column: 1 }, this.ignoreTag = !!r.ignoreTag, this.locale = r.locale, this.requiresOtherClause = !!r.requiresOtherClause, this.shouldParseSkeletons = !!r.shouldParseSkeletons;
    }
    return e.prototype.parse = function() {
      if (this.offset() !== 0)
        throw Error("parser can only be used once");
      return this.parseMessage(0, "", !1);
    }, e.prototype.parseMessage = function(t, r, n) {
      for (var i = []; !this.isEOF(); ) {
        var a = this.char();
        if (a === 123) {
          var o = this.parseArgument(t, n);
          if (o.err)
            return o;
          i.push(o.val);
        } else {
          if (a === 125 && t > 0)
            break;
          if (a === 35 && (r === "plural" || r === "selectordinal")) {
            var u = this.clonePosition();
            this.bump(), i.push({
              type: $.pound,
              location: j(u, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(F.UNMATCHED_CLOSING_TAG, j(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && nn(this.peek() || 0)) {
            var o = this.parseTag(t, r);
            if (o.err)
              return o;
            i.push(o.val);
          } else {
            var o = this.parseLiteral(t, r);
            if (o.err)
              return o;
            i.push(o.val);
          }
        }
      }
      return { val: i, err: null };
    }, e.prototype.parseTag = function(t, r) {
      var n = this.clonePosition();
      this.bump();
      var i = this.parseTagName();
      if (this.bumpSpace(), this.bumpIf("/>"))
        return {
          val: {
            type: $.literal,
            value: "<".concat(i, "/>"),
            location: j(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, u = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !nn(this.char()))
            return this.error(F.INVALID_TAG, j(u, this.clonePosition()));
          var c = this.clonePosition(), f = this.parseTagName();
          return i !== f ? this.error(F.UNMATCHED_CLOSING_TAG, j(c, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: $.tag,
              value: i,
              children: o,
              location: j(n, this.clonePosition())
            },
            err: null
          } : this.error(F.INVALID_TAG, j(u, this.clonePosition())));
        } else
          return this.error(F.UNCLOSED_TAG, j(n, this.clonePosition()));
      } else
        return this.error(F.INVALID_TAG, j(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && oo(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), i = ""; ; ) {
        var a = this.tryParseQuote(r);
        if (a) {
          i += a;
          continue;
        }
        var o = this.tryParseUnquoted(t, r);
        if (o) {
          i += o;
          continue;
        }
        var u = this.tryParseLeftAngleBracket();
        if (u) {
          i += u;
          continue;
        }
        break;
      }
      var c = j(n, this.clonePosition());
      return {
        val: { type: $.literal, value: i, location: c },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !so(this.peek() || 0)) ? (this.bump(), "<") : null;
    }, e.prototype.tryParseQuote = function(t) {
      if (this.isEOF() || this.char() !== 39)
        return null;
      switch (this.peek()) {
        case 39:
          return this.bump(), this.bump(), "'";
        // '{', '<', '>', '}'
        case 123:
        case 60:
        case 62:
        case 125:
          break;
        case 35:
          if (t === "plural" || t === "selectordinal")
            break;
          return null;
        default:
          return null;
      }
      this.bump();
      var r = [this.char()];
      for (this.bump(); !this.isEOF(); ) {
        var n = this.char();
        if (n === 39)
          if (this.peek() === 39)
            r.push(39), this.bump();
          else {
            this.bump();
            break;
          }
        else
          r.push(n);
        this.bump();
      }
      return tn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), tn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(F.EXPECT_ARGUMENT_CLOSING_BRACE, j(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(F.EMPTY_ARGUMENT, j(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(F.MALFORMED_ARGUMENT, j(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(F.EXPECT_ARGUMENT_CLOSING_BRACE, j(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: $.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: j(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(F.EXPECT_ARGUMENT_CLOSING_BRACE, j(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(F.MALFORMED_ARGUMENT, j(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = rn(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = j(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), u = this.parseIdentifierIfPossible().value, c = this.clonePosition();
      switch (u) {
        case "":
          return this.error(F.EXPECT_ARGUMENT_TYPE, j(o, c));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var f = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var d = this.clonePosition(), b = this.parseSimpleArgStyleIfPossible();
            if (b.err)
              return b;
            var m = io(b.val);
            if (m.length === 0)
              return this.error(F.EXPECT_ARGUMENT_STYLE, j(this.clonePosition(), this.clonePosition()));
            var T = j(d, this.clonePosition());
            f = { style: m, styleLocation: T };
          }
          var M = this.tryParseArgumentClose(i);
          if (M.err)
            return M;
          var S = j(i, this.clonePosition());
          if (f && Un(f?.style, "::", 0)) {
            var y = no(f.style.slice(2));
            if (u === "number") {
              var b = this.parseNumberSkeletonFromString(y, f.styleLocation);
              return b.err ? b : {
                val: { type: $.number, value: n, location: S, style: b.val },
                err: null
              };
            } else {
              if (y.length === 0)
                return this.error(F.EXPECT_DATE_TIME_SKELETON, S);
              var v = y;
              this.locale && (v = zs(y, this.locale));
              var m = {
                type: Nt.dateTime,
                pattern: v,
                location: f.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? ks(v) : {}
              }, p = u === "date" ? $.date : $.time;
              return {
                val: { type: p, value: n, location: S, style: m },
                err: null
              };
            }
          }
          return {
            val: {
              type: u === "number" ? $.number : u === "date" ? $.date : $.time,
              value: n,
              location: S,
              style: (a = f?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var g = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(F.EXPECT_SELECT_ARGUMENT_OPTIONS, j(g, z({}, g)));
          this.bumpSpace();
          var x = this.parseIdentifierIfPossible(), E = 0;
          if (u !== "select" && x.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(F.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, j(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var b = this.tryParseDecimalInteger(F.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, F.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (b.err)
              return b;
            this.bumpSpace(), x = this.parseIdentifierIfPossible(), E = b.val;
          }
          var P = this.tryParsePluralOrSelectOptions(t, u, r, x);
          if (P.err)
            return P;
          var M = this.tryParseArgumentClose(i);
          if (M.err)
            return M;
          var N = j(i, this.clonePosition());
          return u === "select" ? {
            val: {
              type: $.select,
              value: n,
              options: Fn(P.val),
              location: N
            },
            err: null
          } : {
            val: {
              type: $.plural,
              value: n,
              options: Fn(P.val),
              offset: E,
              pluralType: u === "plural" ? "cardinal" : "ordinal",
              location: N
            },
            err: null
          };
        }
        default:
          return this.error(F.INVALID_ARGUMENT_TYPE, j(o, c));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(F.EXPECT_ARGUMENT_CLOSING_BRACE, j(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(F.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, j(i, this.clonePosition()));
            this.bump();
            break;
          }
          case 123: {
            t += 1, this.bump();
            break;
          }
          case 125: {
            if (t > 0)
              t -= 1;
            else
              return {
                val: this.message.slice(r.offset, this.offset()),
                err: null
              };
            break;
          }
          default:
            this.bump();
            break;
        }
      }
      return {
        val: this.message.slice(r.offset, this.offset()),
        err: null
      };
    }, e.prototype.parseNumberSkeletonFromString = function(t, r) {
      var n = [];
      try {
        n = Gs(t);
      } catch {
        return this.error(F.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: Nt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Vs(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, u = [], c = /* @__PURE__ */ new Set(), f = i.value, d = i.location; ; ) {
        if (f.length === 0) {
          var b = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var m = this.tryParseDecimalInteger(F.EXPECT_PLURAL_ARGUMENT_SELECTOR, F.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (m.err)
              return m;
            d = j(b, this.clonePosition()), f = this.message.slice(b.offset, this.offset());
          } else
            break;
        }
        if (c.has(f))
          return this.error(r === "select" ? F.DUPLICATE_SELECT_ARGUMENT_SELECTOR : F.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, d);
        f === "other" && (o = !0), this.bumpSpace();
        var T = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? F.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : F.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, j(this.clonePosition(), this.clonePosition()));
        var M = this.parseMessage(t + 1, r, n);
        if (M.err)
          return M;
        var S = this.tryParseArgumentClose(T);
        if (S.err)
          return S;
        u.push([
          f,
          {
            value: M.val,
            location: j(T, this.clonePosition())
          }
        ]), c.add(f), this.bumpSpace(), a = this.parseIdentifierIfPossible(), f = a.value, d = a.location;
      }
      return u.length === 0 ? this.error(r === "select" ? F.EXPECT_SELECT_ARGUMENT_SELECTOR : F.EXPECT_PLURAL_ARGUMENT_SELECTOR, j(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(F.MISSING_OTHER_CLAUSE, j(this.clonePosition(), this.clonePosition())) : { val: u, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var a = !1, o = 0; !this.isEOF(); ) {
        var u = this.char();
        if (u >= 48 && u <= 57)
          a = !0, o = o * 10 + (u - 48), this.bump();
        else
          break;
      }
      var c = j(i, this.clonePosition());
      return a ? (o *= n, to(o) ? { val: o, err: null } : this.error(r, c)) : this.error(t, c);
    }, e.prototype.offset = function() {
      return this.position.offset;
    }, e.prototype.isEOF = function() {
      return this.offset() === this.message.length;
    }, e.prototype.clonePosition = function() {
      return {
        offset: this.position.offset,
        line: this.position.line,
        column: this.position.column
      };
    }, e.prototype.char = function() {
      var t = this.position.offset;
      if (t >= this.message.length)
        throw Error("out of bound");
      var r = Ci(this.message, t);
      if (r === void 0)
        throw Error("Offset ".concat(t, " is at invalid UTF-16 code unit boundary"));
      return r;
    }, e.prototype.error = function(t, r) {
      return {
        val: null,
        err: {
          kind: t,
          message: this.message,
          location: r
        }
      };
    }, e.prototype.bump = function() {
      if (!this.isEOF()) {
        var t = this.char();
        t === 10 ? (this.position.line += 1, this.position.column = 1, this.position.offset += 1) : (this.position.column += 1, this.position.offset += t < 65536 ? 1 : 2);
      }
    }, e.prototype.bumpIf = function(t) {
      if (Un(this.message, t, this.offset())) {
        for (var r = 0; r < t.length; r++)
          this.bump();
        return !0;
      }
      return !1;
    }, e.prototype.bumpUntil = function(t) {
      var r = this.offset(), n = this.message.indexOf(t, r);
      return n >= 0 ? (this.bumpTo(n), !0) : (this.bumpTo(this.message.length), !1);
    }, e.prototype.bumpTo = function(t) {
      if (this.offset() > t)
        throw Error("targetOffset ".concat(t, " must be greater than or equal to the current offset ").concat(this.offset()));
      for (t = Math.min(t, this.message.length); ; ) {
        var r = this.offset();
        if (r === t)
          break;
        if (r > t)
          throw Error("targetOffset ".concat(t, " is at invalid UTF-16 code unit boundary"));
        if (this.bump(), this.isEOF())
          break;
      }
    }, e.prototype.bumpSpace = function() {
      for (; !this.isEOF() && ki(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function nn(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function so(e) {
  return nn(e) || e === 47;
}
function oo(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function ki(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function lo(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function an(e) {
  e.forEach(function(t) {
    if (delete t.location, Hi(t) || Mi(t))
      for (var r in t.options)
        delete t.options[r].location, an(t.options[r].value);
    else Ti(t) && Pi(t.style) || (Si(t) || Ai(t)) && $r(t.style) ? delete t.style.location : Oi(t) && an(t.children);
  });
}
function uo(e, t) {
  t === void 0 && (t = {}), t = z({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new ao(e, t).parse();
  if (r.err) {
    var n = SyntaxError(F[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || an(r.val), r.val;
}
var It;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(It || (It = {}));
var Er = (
  /** @class */
  (function(e) {
    xr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Vn = (
  /** @class */
  (function(e) {
    xr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), It.INVALID_VALUE, a) || this;
    }
    return t;
  })(Er)
), fo = (
  /** @class */
  (function(e) {
    xr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), It.INVALID_VALUE, i) || this;
    }
    return t;
  })(Er)
), co = (
  /** @class */
  (function(e) {
    xr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), It.MISSING_VALUE, n) || this;
    }
    return t;
  })(Er)
), Ee;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(Ee || (Ee = {}));
function ho(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== Ee.literal || r.type !== Ee.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function vo(e) {
  return typeof e == "function";
}
function lr(e, t, r, n, i, a, o) {
  if (e.length === 1 && Rn(e[0]))
    return [
      {
        type: Ee.literal,
        value: e[0].value
      }
    ];
  for (var u = [], c = 0, f = e; c < f.length; c++) {
    var d = f[c];
    if (Rn(d)) {
      u.push({
        type: Ee.literal,
        value: d.value
      });
      continue;
    }
    if (Cs(d)) {
      typeof a == "number" && u.push({
        type: Ee.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var b = d.value;
    if (!(i && b in i))
      throw new co(b, o);
    var m = i[b];
    if (Ls(d)) {
      (!m || typeof m == "string" || typeof m == "number") && (m = typeof m == "string" || typeof m == "number" ? String(m) : ""), u.push({
        type: typeof m == "string" ? Ee.literal : Ee.object,
        value: m
      });
      continue;
    }
    if (Si(d)) {
      var T = typeof d.style == "string" ? n.date[d.style] : $r(d.style) ? d.style.parsedOptions : void 0;
      u.push({
        type: Ee.literal,
        value: r.getDateTimeFormat(t, T).format(m)
      });
      continue;
    }
    if (Ai(d)) {
      var T = typeof d.style == "string" ? n.time[d.style] : $r(d.style) ? d.style.parsedOptions : n.time.medium;
      u.push({
        type: Ee.literal,
        value: r.getDateTimeFormat(t, T).format(m)
      });
      continue;
    }
    if (Ti(d)) {
      var T = typeof d.style == "string" ? n.number[d.style] : Pi(d.style) ? d.style.parsedOptions : void 0;
      T && T.scale && (m = m * (T.scale || 1)), u.push({
        type: Ee.literal,
        value: r.getNumberFormat(t, T).format(m)
      });
      continue;
    }
    if (Oi(d)) {
      var M = d.children, S = d.value, y = i[S];
      if (!vo(y))
        throw new fo(S, "function", o);
      var v = lr(M, t, r, n, i, a), p = y(v.map(function(E) {
        return E.value;
      }));
      Array.isArray(p) || (p = [p]), u.push.apply(u, p.map(function(E) {
        return {
          type: typeof E == "string" ? Ee.literal : Ee.object,
          value: E
        };
      }));
    }
    if (Hi(d)) {
      var g = d.options[m] || d.options.other;
      if (!g)
        throw new Vn(d.value, m, Object.keys(d.options), o);
      u.push.apply(u, lr(g.value, t, r, n, i));
      continue;
    }
    if (Mi(d)) {
      var g = d.options["=".concat(m)];
      if (!g) {
        if (!Intl.PluralRules)
          throw new Er(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, It.MISSING_INTL_API, o);
        var x = r.getPluralRules(t, { type: d.pluralType }).select(m - (d.offset || 0));
        g = d.options[x] || d.options.other;
      }
      if (!g)
        throw new Vn(d.value, m, Object.keys(d.options), o);
      u.push.apply(u, lr(g.value, t, r, n, i, m - (d.offset || 0)));
      continue;
    }
  }
  return ho(u);
}
function po(e, t) {
  return t ? z(z(z({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = z(z({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function mo(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = po(e[n], t[n]), r;
  }, z({}, e)) : e;
}
function Ur(e) {
  return {
    create: function() {
      return {
        get: function(t) {
          return e[t];
        },
        set: function(t, r) {
          e[t] = r;
        }
      };
    }
  };
}
function go(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Rr([void 0], r, !1)))();
    }, {
      cache: Ur(e.number),
      strategy: Dr.variadic
    }),
    getDateTimeFormat: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Rr([void 0], r, !1)))();
    }, {
      cache: Ur(e.dateTime),
      strategy: Dr.variadic
    }),
    getPluralRules: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Rr([void 0], r, !1)))();
    }, {
      cache: Ur(e.pluralRules),
      strategy: Dr.variadic
    })
  };
}
var bo = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var a = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(c) {
        var f = a.formatToParts(c);
        if (f.length === 1)
          return f[0].value;
        var d = f.reduce(function(b, m) {
          return !b.length || m.type !== Ee.literal || typeof b[b.length - 1] != "string" ? b.push(m.value) : b[b.length - 1] += m.value, b;
        }, []);
        return d.length <= 1 ? d[0] || "" : d;
      }, this.formatToParts = function(c) {
        return lr(a.ast, a.locales, a.formatters, a.formats, c, void 0, a.message);
      }, this.resolvedOptions = function() {
        var c;
        return {
          locale: ((c = a.resolvedLocale) === null || c === void 0 ? void 0 : c.toString()) || Intl.NumberFormat.supportedLocalesOf(a.locales)[0]
        };
      }, this.getAst = function() {
        return a.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var o = i || {};
        o.formatters;
        var u = As(o, ["formatters"]);
        this.ast = e.__parse(t, z(z({}, u), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = mo(e.formats, n), this.formatters = i && i.formatters || go(this.formatterCache);
    }
    return Object.defineProperty(e, "defaultLocale", {
      get: function() {
        return e.memoizedDefaultLocale || (e.memoizedDefaultLocale = new Intl.NumberFormat().resolvedOptions().locale), e.memoizedDefaultLocale;
      },
      enumerable: !1,
      configurable: !0
    }), e.memoizedDefaultLocale = null, e.resolveLocale = function(t) {
      if (!(typeof Intl.Locale > "u")) {
        var r = Intl.NumberFormat.supportedLocalesOf(t);
        return r.length > 0 ? new Intl.Locale(r[0]) : new Intl.Locale(typeof t == "string" ? t : t[0]);
      }
    }, e.__parse = uo, e.formats = {
      number: {
        integer: {
          maximumFractionDigits: 0
        },
        currency: {
          style: "currency"
        },
        percent: {
          style: "percent"
        }
      },
      date: {
        short: {
          month: "numeric",
          day: "numeric",
          year: "2-digit"
        },
        medium: {
          month: "short",
          day: "numeric",
          year: "numeric"
        },
        long: {
          month: "long",
          day: "numeric",
          year: "numeric"
        },
        full: {
          weekday: "long",
          month: "long",
          day: "numeric",
          year: "numeric"
        }
      },
      time: {
        short: {
          hour: "numeric",
          minute: "numeric"
        },
        medium: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric"
        },
        long: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        },
        full: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        }
      }
    }, e;
  })()
);
function _o(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let i = 0; i < r.length; i++)
    if (typeof n == "object") {
      if (i > 0) {
        const a = r.slice(i, r.length).join(".");
        if (a in n) {
          n = n[a];
          break;
        }
      }
      n = n[r[i]];
    } else
      n = void 0;
  return n;
}
const lt = {}, yo = (e, t, r) => r && (t in lt || (lt[t] = {}), e in lt[t] || (lt[t][e] = r), r), Di = (e, t) => {
  if (t == null)
    return;
  if (t in lt && e in lt[t])
    return lt[t][e];
  const r = wr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = Eo(i, e);
    if (a)
      return yo(e, t, a);
  }
};
let bn;
const Yt = Zt({});
function xo(e) {
  return bn[e] || null;
}
function Gi(e) {
  return e in bn;
}
function Eo(e, t) {
  if (!Gi(e))
    return null;
  const r = xo(e);
  return _o(r, t);
}
function wo(e) {
  if (e == null)
    return;
  const t = wr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Gi(n))
      return n;
  }
}
function To(e, ...t) {
  delete lt[e], Yt.update((r) => (r[e] = Ss.all([r[e] || {}, ...t]), r));
}
Lt(
  [Yt],
  ([e]) => Object.keys(e)
);
Yt.subscribe((e) => bn = e);
const ur = {};
function So(e, t) {
  ur[e].delete(t), ur[e].size === 0 && delete ur[e];
}
function Ui(e) {
  return ur[e];
}
function Ao(e) {
  return wr(e).map((t) => {
    const r = Ui(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function sn(e) {
  return e == null ? !1 : wr(e).some(
    (t) => {
      var r;
      return (r = Ui(t)) == null ? void 0 : r.size;
    }
  );
}
function Ho(e, t) {
  return Promise.all(
    t.map((n) => (So(e, n), n().then((i) => i.default || i)))
  ).then((n) => To(e, ...n));
}
const jt = {};
function Fi(e) {
  if (!sn(e))
    return e in jt ? jt[e] : Promise.resolve();
  const t = Ao(e);
  return jt[e] = Promise.all(
    t.map(
      ([r, n]) => Ho(r, n)
    )
  ).then(() => {
    if (sn(e))
      return Fi(e);
    delete jt[e];
  }), jt[e];
}
const Mo = {
  number: {
    scientific: { notation: "scientific" },
    engineering: { notation: "engineering" },
    compactLong: { notation: "compact", compactDisplay: "long" },
    compactShort: { notation: "compact", compactDisplay: "short" }
  },
  date: {
    short: { month: "numeric", day: "numeric", year: "2-digit" },
    medium: { month: "short", day: "numeric", year: "numeric" },
    long: { month: "long", day: "numeric", year: "numeric" },
    full: { weekday: "long", month: "long", day: "numeric", year: "numeric" }
  },
  time: {
    short: { hour: "numeric", minute: "numeric" },
    medium: { hour: "numeric", minute: "numeric", second: "numeric" },
    long: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    },
    full: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    }
  }
}, Oo = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: Mo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, Po = Oo;
function Bt() {
  return Po;
}
const Fr = Zt(!1);
var No = Object.defineProperty, Io = Object.defineProperties, Bo = Object.getOwnPropertyDescriptors, zn = Object.getOwnPropertySymbols, Lo = Object.prototype.hasOwnProperty, Co = Object.prototype.propertyIsEnumerable, Xn = (e, t, r) => t in e ? No(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Ro = (e, t) => {
  for (var r in t || (t = {}))
    Lo.call(t, r) && Xn(e, r, t[r]);
  if (zn)
    for (var r of zn(t))
      Co.call(t, r) && Xn(e, r, t[r]);
  return e;
}, ko = (e, t) => Io(e, Bo(t));
let on;
const dr = Zt(null);
function qn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function wr(e, t = Bt().fallbackLocale) {
  const r = qn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...qn(t)])] : r;
}
function Tt() {
  return on ?? void 0;
}
dr.subscribe((e) => {
  on = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const Do = (e) => {
  if (e && wo(e) && sn(e)) {
    const { loadingDelay: t } = Bt();
    let r;
    return typeof window < "u" && Tt() != null && t ? r = window.setTimeout(
      () => Fr.set(!0),
      t
    ) : Fr.set(!0), Fi(e).then(() => {
      dr.set(e);
    }).finally(() => {
      clearTimeout(r), Fr.set(!1);
    });
  }
  return dr.set(e);
}, Ct = ko(Ro({}, dr), {
  set: Do
}), Tr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Go = Object.defineProperty, vr = Object.getOwnPropertySymbols, ji = Object.prototype.hasOwnProperty, Vi = Object.prototype.propertyIsEnumerable, Wn = (e, t, r) => t in e ? Go(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, _n = (e, t) => {
  for (var r in t || (t = {}))
    ji.call(t, r) && Wn(e, r, t[r]);
  if (vr)
    for (var r of vr(t))
      Vi.call(t, r) && Wn(e, r, t[r]);
  return e;
}, Rt = (e, t) => {
  var r = {};
  for (var n in e)
    ji.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && vr)
    for (var n of vr(e))
      t.indexOf(n) < 0 && Vi.call(e, n) && (r[n] = e[n]);
  return r;
};
const qt = (e, t) => {
  const { formats: r } = Bt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Uo = Tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Rt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = qt("number", n)), new Intl.NumberFormat(r, i);
  }
), Fo = Tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Rt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = qt("date", n) : Object.keys(i).length === 0 && (i = qt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), jo = Tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Rt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = qt("time", n) : Object.keys(i).length === 0 && (i = qt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Vo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = Rt(t, [
    "locale"
  ]);
  return Uo(_n({ locale: r }, n));
}, zo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = Rt(t, [
    "locale"
  ]);
  return Fo(_n({ locale: r }, n));
}, Xo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = Rt(t, [
    "locale"
  ]);
  return jo(_n({ locale: r }, n));
}, qo = Tr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Tt()) => new bo(e, t, Bt().formats, {
    ignoreTag: Bt().ignoreTag
  })
), Wo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: u,
    locale: c = Tt(),
    default: f
  } = o;
  if (c == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let d = Di(e, c);
  if (!d)
    d = (a = (i = (n = (r = Bt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: c, id: e, defaultValue: f })) != null ? i : f) != null ? a : e;
  else if (typeof d != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof d}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), d;
  if (!u)
    return d;
  let b = d;
  try {
    b = qo(d, c).format(u);
  } catch (m) {
    m instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      m.message
    );
  }
  return b;
}, Zo = (e, t) => Xo(t).format(e), Yo = (e, t) => zo(t).format(e), Jo = (e, t) => Vo(t).format(e), Qo = (e, t = Tt()) => Di(e, t);
Lt([Ct, Yt], () => Wo);
Lt([Ct], () => Zo);
Lt([Ct], () => Yo);
Lt([Ct], () => Jo);
Lt([Ct, Yt], () => Qo);
const Ko = "__i18n__", $o = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], el = [
  "elem_id",
  "elem_classes",
  "visible",
  "interactive",
  "server_fns",
  "server",
  "id",
  "target",
  "theme_mode",
  "version",
  "root",
  "autoscroll",
  "max_file_size",
  "formatter",
  "client",
  "load_component",
  "scale",
  "min_width",
  "theme",
  "padding",
  "loading_status",
  "label",
  "show_label",
  "validation_error",
  "show_progress",
  "api_prefix",
  "container",
  "attached_events",
  "register_component",
  "dispatcher"
];
function tl(e) {
  return typeof e == "string" && e.includes(Ko);
}
class rl {
  load_component;
  #t = Q(Xt({}));
  get shared() {
    return l(this.#t);
  }
  set shared(t) {
    H(this.#t, t, !0);
  }
  #r = Q(Xt({}));
  get props() {
    return l(this.#r);
  }
  set props(t) {
    H(this.#r, t, !0);
  }
  #e = Q((t) => t);
  get i18n() {
    return l(this.#e);
  }
  set i18n(t) {
    H(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = el;
  mounted = !1;
  old_value;
  register_component;
  constructor(t, r) {
    for (const n in t.shared_props)
      this.shared[n] = t.shared_props[n];
    for (const n in t.props)
      this.props[n] = t.props[n];
    if (r)
      for (const n in r)
        this.props[n] === void 0 && (this.props[n] = r[n]);
    this.i18n = this.props.i18n ?? ((n) => n);
    for (const n of $o)
      this.shared[n] = this._translate_and_store(
        "shared",
        n,
        // @ts-ignore
        t.shared_props[n]
      ), this.props[n] = this._translate_and_store(
        "props",
        n,
        // @ts-ignore
        t.props[n]
      );
    this.load_component = this.shared.load_component, this.register_component = this.shared.register_component || (() => {
    }), this.dispatcher = this.shared.dispatcher || (() => {
    }), this.register_component(
      t.shared_props.id,
      // @ts-ignore
      this.set_data.bind(this),
      this.get_data.bind(this)
    ), Le(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), he(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Ct.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [a, o] = n.split("."), u = this.i18n(i);
        a === "shared" ? this.shared[o] = u : this.props[o] = u;
      }
    });
  }
  // check if props are translatable
  _is_i18n_managed(t, r) {
    const n = this.translatable_props[t];
    return n ? r === n ? !0 : (delete this.translatable_props[t], !1) : !1;
  }
  _translate_and_store(t, r, n) {
    if (typeof n != "string") return n;
    const i = this.i18n(n);
    return i !== n && (this.translatable_props[`${t}.${r}`] = n), i;
  }
  dispatch(t, r) {
    this.dispatcher(this.shared.id, t, r);
  }
  async get_data() {
    return Ga(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = tl(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    Le(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
La();
var nl = /* @__PURE__ */ vi('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Zn = /* @__PURE__ */ de("<!> <!>", 1), il = /* @__PURE__ */ de('<div class="placeholder svelte-1stq1b1"></div>');
function al(e, t) {
  _r(t, !1);
  let r = L(t, "height", 8, void 0), n = L(t, "min_height", 8, void 0), i = L(t, "max_height", 8, void 0), a = L(t, "width", 8, void 0), o = L(t, "elem_id", 8, ""), u = L(t, "elem_classes", 24, () => []), c = L(t, "variant", 8, "solid"), f = L(t, "border_mode", 8, "base"), d = L(t, "padding", 8, !0), b = L(t, "type", 8, "normal"), m = L(t, "test_id", 8, void 0), T = L(t, "explicit_call", 8, !1), M = L(t, "container", 8, !0), S = L(t, "visible", 8, !0), y = L(t, "allow_overflow", 8, !0), v = L(t, "overflow_behavior", 8, "auto"), p = L(t, "scale", 8, null), g = L(t, "min_width", 8, 0), x = L(t, "flex", 12, !1), E = L(t, "resizable", 8, !1), P = L(t, "rtl", 8, !1), N = L(t, "fullscreen", 12, !1), k = L(t, "label", 8, void 0), D = yt(N()), C = yt(), le = b() === "fieldset" ? "fieldset" : "div", be = yt(0), me = yt(0), K = yt(null);
  function ve(ie) {
    N() && ie.key === "Escape" && N(!1);
  }
  const pe = (ie) => {
    if (ie !== void 0) {
      if (typeof ie == "number")
        return ie + "px";
      if (typeof ie == "string")
        return ie;
    }
  }, ct = (ie) => {
    let V = ie.clientY;
    const Se = (Z) => {
      const se = Z.clientY - V;
      V = Z.clientY, Ra(C, l(C).style.height = `${l(C).offsetHeight + se}px`);
    }, fe = () => {
      window.removeEventListener("mousemove", Se), window.removeEventListener("mouseup", fe);
    };
    window.addEventListener("mousemove", Se), window.addEventListener("mouseup", fe);
  };
  An(
    () => (Ne(N()), l(D), l(C)),
    () => {
      N() !== l(D) && (H(D, N()), N() ? (H(K, l(C).getBoundingClientRect()), H(be, l(C).offsetHeight), H(me, l(C).offsetWidth), window.addEventListener("keydown", ve)) : (H(K, null), window.removeEventListener("keydown", ve)));
    }
  ), An(() => Ne(S()), () => {
    S() || x(!1);
  }), Ca(), ms();
  var we = Ht(), Ge = xe(we);
  {
    var Te = (ie) => {
      var V = Zn(), Se = xe(V);
      is(Se, () => le, !1, (se, Ke) => {
        gn(se, (Ae) => H(C, Ae), () => l(C)), ps(
          se,
          (Ae, qe) => ({
            "data-testid": m(),
            id: o(),
            class: `block ${Ae ?? ""}`,
            dir: P() ? "rtl" : "ltr",
            "aria-label": k(),
            style: "",
            [zt]: {
              hidden: S() === "hidden",
              padded: d(),
              flex: x(),
              border_focus: f() === "focus",
              border_contrast: f() === "contrast",
              "hide-container": !T() && !M(),
              fullscreen: N(),
              animating: N() && l(K) !== null,
              "auto-margin": p() === null
            },
            [At]: qe
          }),
          [
            () => (Ne(u()), he(() => u()?.join(" ") || "")),
            () => ({
              height: (Ne(N()), Ne(r()), he(() => N() ? void 0 : pe(r()))),
              "min-height": (Ne(N()), Ne(n()), he(() => N() ? void 0 : pe(n()))),
              "max-height": (Ne(N()), Ne(i()), he(() => N() ? void 0 : pe(i()))),
              "--start-top": (l(K), he(() => l(K) ? `${l(K).top}px` : "0px")),
              "--start-left": (l(K), he(() => l(K) ? `${l(K).left}px` : "0px")),
              "--start-width": (l(K), he(() => l(K) ? `${l(K).width}px` : "0px")),
              "--start-height": (l(K), he(() => l(K) ? `${l(K).height}px` : "0px")),
              width: (Ne(N()), Ne(a()), he(() => N() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : pe(a()))),
              "border-style": c(),
              overflow: y() ? v() : "hidden",
              "flex-grow": p(),
              "min-width": `calc(min(${g()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Xe = Zn(), ht = xe(Xe);
        Qr(ht, t, "default", {});
        var Ue = W(ht, 2);
        {
          var $e = (Ae) => {
            var qe = nl();
            Ie("mousedown", qe, ct), U(Ae, qe);
          };
          re(Ue, (Ae) => {
            E() && Ae($e);
          });
        }
        U(Ke, Xe);
      });
      var fe = W(Se, 2);
      {
        var Z = (se) => {
          var Ke = il();
          let Xe;
          ne(() => Xe = De(Ke, "", Xe, {
            height: l(be) + "px",
            width: l(me) + "px"
          })), U(se, Ke);
        };
        re(fe, (se) => {
          N() && se(Z);
        });
      }
      U(ie, V);
    };
    re(Ge, (ie) => {
      (S() === !0 || S() === "hidden") && ie(Te);
    });
  }
  U(e, we), br();
}
var sl = /* @__PURE__ */ de('<span class="svelte-vvirtv"> </span>'), ol = /* @__PURE__ */ de("<button><!> <div><!> <!></div></button>");
function Yn(e, t) {
  let r = L(t, "label", 3, ""), n = L(t, "show_label", 3, !1), i = L(t, "pending", 3, !1), a = L(t, "size", 3, "small"), o = L(t, "padded", 3, !0), u = L(t, "highlight", 3, !1), c = L(t, "disabled", 3, !1), f = L(t, "hasPopup", 3, !1), d = L(t, "color", 3, "var(--block-label-text-color)"), b = L(t, "transparent", 3, !1), m = L(t, "background", 3, "var(--block-background-fill)"), T = L(t, "border", 3, "transparent"), M = Be(() => u() ? "var(--color-accent)" : d());
  var S = ol();
  let y, v;
  var p = ae(S);
  {
    var g = (D) => {
      var C = sl(), le = ae(C);
      ne(() => ge(le, r())), U(D, C);
    };
    re(p, (D) => {
      n() && D(g);
    });
  }
  var x = W(p, 2);
  let E;
  var P = ae(x);
  ts(P, () => t.Icon, (D, C) => {
    C(D, {});
  });
  var N = W(P, 2);
  {
    var k = (D) => {
      var C = Ht(), le = xe(C);
      Za(le, () => t.children), U(D, C);
    };
    re(N, (D) => {
      t.children && D(k);
    });
  }
  ne(() => {
    y = Et(S, 1, "icon-button svelte-vvirtv", null, y, {
      pending: i(),
      padded: o(),
      highlight: u(),
      transparent: b()
    }), S.disabled = c(), Ot(S, "aria-label", r()), Ot(S, "aria-haspopup", f()), Ot(S, "title", r()), v = De(S, "", v, {
      "--border-color": T(),
      color: !c() && l(M) ? l(M) : "var(--block-label-text-color)",
      "--bg-color": c() ? "auto" : m()
    }), E = Et(x, 1, "svelte-vvirtv", null, E, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), fi("click", S, function(...D) {
    t.onclick?.apply(this, D);
  }), U(e, S);
}
gr(["click"]);
var ll = /* @__PURE__ */ vi('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Jn(e) {
  var t = ll();
  U(e, t);
}
gr(["click"]);
function jr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Qn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function ln(e, t, r, n) {
  if (typeof r == "number" || Qn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, u = e.opts.damping * a, c = (o - u) * e.inv_mass, f = (a + c) * e.dt;
    return Math.abs(f) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Qn(r) ? new Date(r.getTime() + f) : r + f);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          ln(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = ln(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Kn(e, t = {}) {
  const r = Zt(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, u, c, f = (
    /** @type {T} */
    e
  ), d = (
    /** @type {T | undefined} */
    e
  ), b = 1, m = 0, T = !1;
  function M(y, v = {}) {
    d = y;
    const p = c = {};
    return e == null || v.hard || S.stiffness >= 1 && S.damping >= 1 ? (T = !0, o = ke.now(), f = y, r.set(e = d), Promise.resolve()) : (v.soft && (m = 1 / ((v.soft === !0 ? 0.5 : +v.soft) * 60), b = 0), u || (o = ke.now(), T = !1, u = ns((g) => {
      if (T)
        return T = !1, u = null, !1;
      b = Math.min(b + m, 1);
      const x = Math.min(g - o, 1e3 / 30), E = {
        inv_mass: b,
        opts: S,
        settled: !0,
        dt: x * 60 / 1e3
      }, P = ln(E, f, e, d);
      return o = g, f = /** @type {T} */
      e, r.set(e = /** @type {T} */
      P), E.settled && (u = null), !E.settled;
    })), new Promise((g) => {
      u.promise.then(() => {
        p === c && g();
      });
    }));
  }
  const S = {
    set: M,
    update: (y, v) => M(y(
      /** @type {T} */
      d,
      /** @type {T} */
      e
    ), v),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return S;
}
var ul = /* @__PURE__ */ de('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function fl(e, t) {
  _r(t, !0);
  const r = () => Hn(c, "$top", i), n = () => Hn(f, "$bottom", i), [i, a] = ja();
  var o = this && this.__awaiter || function(g, x, E, P) {
    function N(k) {
      return k instanceof E ? k : new E(function(D) {
        D(k);
      });
    }
    return new (E || (E = Promise))(function(k, D) {
      function C(me) {
        try {
          be(P.next(me));
        } catch (K) {
          D(K);
        }
      }
      function le(me) {
        try {
          be(P.throw(me));
        } catch (K) {
          D(K);
        }
      }
      function be(me) {
        me.done ? k(me.value) : N(me.value).then(C, le);
      }
      be((P = P.apply(g, x || [])).next());
    });
  };
  let u = L(t, "margin", 3, !0);
  const c = Kn([0, 0]), f = Kn([0, 0]);
  let d = Q(!1);
  function b() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 140]), f.set([-125, -140])]), yield Promise.all([c.set([-125, 140]), f.set([125, -140])]), yield Promise.all([c.set([-125, 0]), f.set([125, -0])]), yield Promise.all([c.set([125, 0]), f.set([-125, 0])]);
    });
  }
  function m() {
    return o(this, void 0, void 0, function* () {
      yield b(), l(d) || m();
    });
  }
  function T() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 0]), f.set([-125, 0])]), m();
    });
  }
  Le(() => (T(), () => {
    H(d, !0);
  }));
  var M = ul();
  let S;
  var y = ae(M), v = ae(y), p = W(v);
  ne(() => {
    S = Et(M, 1, "svelte-m6d381", null, S, { margin: u() }), De(v, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), De(p, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), U(e, M), br(), a();
}
var cl = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function u(d) {
      try {
        f(n.next(d));
      } catch (b) {
        o(b);
      }
    }
    function c(d) {
      try {
        f(n.throw(d));
      } catch (b) {
        o(b);
      }
    }
    function f(d) {
      d.done ? a(d.value) : i(d.value).then(u, c);
    }
    f((n = n.apply(e, t || [])).next());
  });
};
let sr = [], Vr = !1;
const hl = typeof window < "u", zi = hl ? window.requestAnimationFrame : (e) => {
};
function dl(e) {
  return cl(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (sr.push(t), !Vr) Vr = !0;
      else return;
      yield ka(), zi(() => {
        let n = [0, 0];
        for (let i = 0; i < sr.length; i++) {
          const o = sr[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), Vr = !1, sr = [];
      });
    }
  });
}
var vl = /* @__PURE__ */ de('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), pl = /* @__PURE__ */ de('<div class="eta-bar svelte-124hqw6"></div>'), ml = /* @__PURE__ */ de("<!> ", 1), gl = /* @__PURE__ */ de("<!> <!> <!> <!>", 1), bl = /* @__PURE__ */ de('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), _l = /* @__PURE__ */ de('<p class="loading svelte-124hqw6"> </p> <!>', 1), yl = /* @__PURE__ */ de("<!> <div><!> <!></div> <!> <!>", 1), xl = /* @__PURE__ */ de('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), El = /* @__PURE__ */ de("<div> <!> </div>"), wl = /* @__PURE__ */ de('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function Tl(e, t) {
  _r(t, !0);
  let r = L(t, "eta", 3, null), n = L(t, "scroll_to_output", 3, !1), i = L(t, "timer", 3, !0), a = L(t, "show_progress", 3, "full"), o = L(t, "message", 3, null), u = L(t, "progress", 3, null), c = L(t, "variant", 3, "default"), f = L(t, "loading_text", 3, "Loading..."), d = L(t, "absolute", 3, !0), b = L(t, "translucent", 3, !1), m = L(t, "border", 3, !1), T = L(t, "validation_error", 7, null), M = L(t, "show_validation_error", 3, !0), S = L(t, "type", 3, null), y = L(t, "used_cache", 3, null), v = L(t, "cache_duration", 3, null), p = L(t, "avg_time", 3, null), g, x = !1, E = Q(0), P = Q(null), N = Q(null), k = Q(!1), D = Q(null), C = Q(!1), le = Q(!1), be = Q(null), me = Q(null), K = Q("from cache"), ve = Q(!1), pe = null, ct = null;
  const we = Be(() => !(M() && T()) && (S() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Ge = Q(0);
  const Te = Be(() => l(N) === null || l(N) <= 0 || !l(Ge) ? 0 : Math.min(l(Ge) / l(N), 1)), ie = Be(() => l(Ge).toFixed(1));
  let V = Be(() => u() == null), Se = Be(() => r() !== null && r() !== void 0 ? r() : l(P));
  function fe() {
    zi(() => {
      H(Ge, (performance.now() - l(E)) / 1e3), x && fe();
    });
  }
  let Z = Be(() => {
    let Y = null;
    u() != null ? Y = u().map((oe) => {
      if (oe.index != null && oe.length != null)
        return oe.index / oe.length;
      if (oe.progress != null)
        return oe.progress;
    }) : Y = null;
    let ee, ce = "";
    return Y ? (ee = Y[Y.length - 1], ee === 0 ? ce = "0" : ce = "150ms") : ee = void 0, {
      progress_level: Y,
      last_progress_level: ee,
      progress_bar_transition: ce
    };
  });
  function se() {
    x || (H(P, H(D, null), !0), H(E, performance.now(), !0), x = !0, fe());
  }
  function Ke() {
    H(P, H(D, null), !0), x && (x = !1);
  }
  Le(() => {
    t.status === "pending" ? se() : he(() => {
      Ke();
    });
  }), Le(() => {
    g && n() && (t.status === "pending" || t.status === "complete") && dl(g, t.autoscroll);
  }), Le(() => {
    l(Se) != null && l(P) !== l(Se) && (H(N, (performance.now() - l(E)) / 1e3 + l(Se)), H(D, l(N).toFixed(1), !0), H(P, l(Se), !0));
  });
  function Xe() {
    H(k, !1);
  }
  Le(() => {
    he(() => {
      Xe();
    }), t.status === "error" && o() && H(k, !0);
  }), Le(() => {
    t.status === "complete" && S() === "output" && y() && v() != null && (H(be, v().toFixed(1), !0), H(K, y() === "full" ? "from cache" : "used cache", !0), H(ve, p() != null && p() > v() && p() > 0, !0), H(me, l(ve) ? p().toFixed(1) : null, !0), H(C, !0), H(le, !1), pe && clearTimeout(pe), ct && clearTimeout(ct), pe = setTimeout(
      () => {
        H(le, !0), ct = setTimeout(
          () => {
            H(C, !1), H(le, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var ht = wl(), Ue = xe(ht);
  let $e, Ae;
  var qe = ae(Ue);
  {
    var Sr = (Y) => {
      var ee = vl(), ce = ae(ee), oe = W(ce), ue = ae(oe);
      {
        let Me = Be(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Yn(ue, {
          get Icon() {
            return Jn;
          },
          get label() {
            return l(Me);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => T(null)
        });
      }
      ne(() => ge(ce, `${T() ?? ""} `)), U(Y, ee);
    };
    re(qe, (Y) => {
      T() && M() && Y(Sr);
    });
  }
  var et = W(qe, 2);
  {
    var Ar = (Y) => {
      var ee = yl(), ce = xe(ee);
      {
        var oe = (q) => {
          var te = pl();
          let He;
          ne(() => He = De(te, "", He, {
            transform: `translateX(${(l(Te) || 0) * 100 - 100}%)`
          })), U(q, te);
        };
        re(ce, (q) => {
          c() === "default" && l(V) && a() === "full" && q(oe);
        });
      }
      var ue = W(ce, 2);
      let Me;
      var We = ae(ue);
      {
        var Ce = (q) => {
          var te = Ht(), He = xe(te);
          Jr(He, 17, u, Zr, (rt, Oe) => {
            var _e = Ht(), ye = xe(_e);
            {
              var nt = (Ze) => {
                var pt = ml(), mt = xe(pt);
                {
                  var Gt = (Re) => {
                    var Je = Ve();
                    ne((gt, bt) => ge(Je, `${gt ?? ""}/${bt ?? ""}`), [
                      () => jr(l(Oe).index || 0),
                      () => jr(l(Oe).length)
                    ]), U(Re, Je);
                  }, it = (Re) => {
                    var Je = Ve();
                    ne((gt) => ge(Je, gt), [() => jr(l(Oe).index || 0)]), U(Re, Je);
                  };
                  re(mt, (Re) => {
                    l(Oe).length != null ? Re(Gt) : Re(it, -1);
                  });
                }
                var Ye = W(mt);
                ne(() => ge(Ye, ` ${l(Oe).unit ?? ""} |  `)), U(Ze, pt);
              };
              re(ye, (Ze) => {
                l(Oe).index != null && Ze(nt);
              });
            }
            U(rt, _e);
          }), U(q, te);
        }, Fe = (q) => {
          var te = Ve();
          ne(() => ge(te, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), U(q, te);
        }, vt = (q) => {
          var te = Ve("processing |");
          U(q, te);
        };
        re(We, (q) => {
          u() ? q(Ce) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? q(Fe, 1) : t.queue_position === 0 && q(vt, 2);
        });
      }
      var Jt = W(We, 2);
      {
        var Qt = (q) => {
          var te = Ve();
          ne(() => ge(te, `${l(ie) ?? ""}${r() ? `/${l(D)}` : ""}s`)), U(q, te);
        };
        re(Jt, (q) => {
          i() && q(Qt);
        });
      }
      var Dt = W(ue, 2);
      {
        var Mr = (q) => {
          var te = bl(), He = ae(te), rt = ae(He);
          {
            var Oe = (Ze) => {
              var pt = Ht(), mt = xe(pt);
              Jr(mt, 17, u, Zr, (Gt, it, Ye) => {
                var Re = Ht(), Je = xe(Re);
                {
                  var gt = (bt) => {
                    var er = gl(), s = xe(er);
                    {
                      var h = (I) => {
                        var G = Ve(" /");
                        U(I, G);
                      };
                      re(s, (I) => {
                        Ye !== 0 && I(h);
                      });
                    }
                    var _ = W(s, 2);
                    {
                      var w = (I) => {
                        var G = Ve();
                        ne(() => ge(G, l(it).desc)), U(I, G);
                      };
                      re(_, (I) => {
                        l(it).desc != null && I(w);
                      });
                    }
                    var O = W(_, 2);
                    {
                      var A = (I) => {
                        var G = Ve("-");
                        U(I, G);
                      };
                      re(O, (I) => {
                        l(it).desc != null && l(Z).progress_level && l(Z).progress_level[Ye] != null && I(A);
                      });
                    }
                    var B = W(O, 2);
                    {
                      var R = (I) => {
                        var G = Ve();
                        ne((X) => ge(G, `${X ?? ""}%`), [
                          () => (100 * (l(Z).progress_level[Ye] || 0)).toFixed(1)
                        ]), U(I, G);
                      };
                      re(B, (I) => {
                        l(Z).progress_level != null && I(R);
                      });
                    }
                    U(bt, er);
                  };
                  re(Je, (bt) => {
                    (l(it).desc != null || l(Z).progress_level && l(Z).progress_level[Ye] != null) && bt(gt);
                  });
                }
                U(Gt, Re);
              }), U(Ze, pt);
            };
            re(rt, (Ze) => {
              u() != null && Ze(Oe);
            });
          }
          var _e = W(He, 2), ye = ae(_e);
          let nt;
          ne(() => nt = De(ye, "", nt, {
            width: `${l(Z).last_progress_level * 100}%`,
            transition: l(Z).progress_bar_transition
          })), U(q, te);
        }, Kt = (q) => {
          {
            let te = Be(() => c() === "default");
            fl(q, {
              get margin() {
                return l(te);
              }
            });
          }
        };
        re(Dt, (q) => {
          l(Z).last_progress_level != null ? q(Mr) : a() === "full" && q(Kt, 1);
        });
      }
      var $t = W(Dt, 2);
      {
        var tt = (q) => {
          var te = _l(), He = xe(te), rt = ae(He), Oe = W(He, 2);
          Qr(Oe, t, "additional-loading-text", {}), ne(() => ge(rt, f())), U(q, te);
        };
        re($t, (q) => {
          i() || q(tt);
        });
      }
      ne(() => Me = Et(ue, 1, "progress-text svelte-124hqw6", null, Me, {
        "meta-text-center": c() === "center",
        "meta-text": c() === "default"
      })), U(Y, ee);
    }, Hr = (Y) => {
      var ee = xl(), ce = xe(ee), oe = ae(ce);
      {
        let Ce = Be(() => t.i18n("common.clear"));
        Yn(oe, {
          get Icon() {
            return Jn;
          },
          get label() {
            return l(Ce);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var ue = W(ce, 2), Me = ae(ue), We = W(ue, 2);
      Qr(We, t, "error", {}), ne((Ce) => ge(Me, Ce), [() => t.i18n("common.error")]), U(Y, ee);
    };
    re(et, (Y) => {
      t.status === "pending" ? Y(Ar) : t.status === "error" && Y(Hr, 1);
    });
  }
  gn(Ue, (Y) => g = Y, () => g);
  var kt = W(Ue, 2);
  {
    var dt = (Y) => {
      var ee = El();
      let ce, oe;
      var ue = ae(ee), Me = W(ue);
      {
        var We = (Fe) => {
          var vt = Ve();
          ne(() => ge(vt, `~${l(me) ?? ""}s
			→ `)), U(Fe, vt);
        };
        re(Me, (Fe) => {
          l(ve) && Fe(We);
        });
      }
      var Ce = W(Me);
      ne(() => {
        ce = Et(ee, 1, "cache-indicator svelte-124hqw6", null, ce, { "fade-out": l(le) }), oe = De(ee, "", oe, { position: d() ? "absolute" : "static" }), ge(ue, `⚡ ${l(K) ?? ""}: `), ge(Ce, `${l(be) ?? ""}s`);
      }), U(Y, ee);
    };
    re(kt, (Y) => {
      l(C) && Y(dt);
    });
  }
  ne(() => {
    $e = Et(Ue, 1, `wrap ${c() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", $e, {
      "no-click": T() && M(),
      hide: l(we),
      translucent: c() === "center" && (t.status === "pending" || t.status === "error") || b() || a() === "minimal" || T(),
      generating: t.status === "generating" && a() === "full",
      border: m()
    }), Ae = De(Ue, "", Ae, {
      position: d() ? "absolute" : "static",
      padding: d() ? "0" : "var(--size-8) 0"
    });
  }), U(e, ht), br();
}
const Sl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, Al = [
  "abbr",
  "accept",
  "accept-charset",
  "accesskey",
  "action",
  "align",
  "alink",
  "allow",
  "allowfullscreen",
  "alt",
  "anchor",
  "archive",
  "as",
  "async",
  "autocapitalize",
  "autocomplete",
  "autocorrect",
  "autofocus",
  "autopictureinpicture",
  "autoplay",
  "axis",
  "background",
  "behavior",
  "bgcolor",
  "border",
  "bordercolor",
  "capture",
  "cellpadding",
  "cellspacing",
  "challenge",
  "char",
  "charoff",
  "charset",
  "checked",
  "cite",
  "class",
  "classid",
  "clear",
  "code",
  "codebase",
  "codetype",
  "color",
  "cols",
  "colspan",
  "compact",
  "content",
  "contenteditable",
  "controls",
  "controlslist",
  "conversiondestination",
  "coords",
  "crossorigin",
  "csp",
  "data",
  "datetime",
  "declare",
  "decoding",
  "default",
  "defer",
  "dir",
  "direction",
  "dirname",
  "disabled",
  "disablepictureinpicture",
  "disableremoteplayback",
  "disallowdocumentaccess",
  "download",
  "draggable",
  "elementtiming",
  "enctype",
  "end",
  "enterkeyhint",
  "event",
  "exportparts",
  "face",
  "for",
  "form",
  "formaction",
  "formenctype",
  "formmethod",
  "formnovalidate",
  "formtarget",
  "frame",
  "frameborder",
  "headers",
  "height",
  "hidden",
  "high",
  "href",
  "hreflang",
  "hreftranslate",
  "hspace",
  "http-equiv",
  "id",
  "imagesizes",
  "imagesrcset",
  "importance",
  "impressiondata",
  "impressionexpiry",
  "incremental",
  "inert",
  "inputmode",
  "integrity",
  "invisible",
  "ismap",
  "keytype",
  "kind",
  "label",
  "lang",
  "language",
  "latencyhint",
  "leftmargin",
  "link",
  "list",
  "loading",
  "longdesc",
  "loop",
  "low",
  "lowsrc",
  "manifest",
  "marginheight",
  "marginwidth",
  "max",
  "maxlength",
  "mayscript",
  "media",
  "method",
  "min",
  "minlength",
  "multiple",
  "muted",
  "name",
  "nohref",
  "nomodule",
  "nonce",
  "noresize",
  "noshade",
  "novalidate",
  "nowrap",
  "object",
  "open",
  "optimum",
  "part",
  "pattern",
  "ping",
  "placeholder",
  "playsinline",
  "policy",
  "poster",
  "preload",
  "pseudo",
  "readonly",
  "referrerpolicy",
  "rel",
  "reportingorigin",
  "required",
  "resources",
  "rev",
  "reversed",
  "role",
  "rows",
  "rowspan",
  "rules",
  "sandbox",
  "scheme",
  "scope",
  "scopes",
  "scrollamount",
  "scrolldelay",
  "scrolling",
  "select",
  "selected",
  "shadowroot",
  "shadowrootdelegatesfocus",
  "shape",
  "size",
  "sizes",
  "slot",
  "span",
  "spellcheck",
  "src",
  "srclang",
  "srcset",
  "standby",
  "start",
  "step",
  "style",
  "summary",
  "tabindex",
  "target",
  "text",
  "title",
  "topmargin",
  "translate",
  "truespeed",
  "trusttoken",
  "type",
  "usemap",
  "valign",
  "value",
  "valuetype",
  "version",
  "virtualkeyboardpolicy",
  "vlink",
  "vspace",
  "webkitdirectory",
  "width",
  "wrap"
], Hl = [
  "accent-height",
  "accumulate",
  "additive",
  "alignment-baseline",
  "ascent",
  "attributename",
  "attributetype",
  "azimuth",
  "basefrequency",
  "baseline-shift",
  "begin",
  "bias",
  "by",
  "class",
  "clip",
  "clippathunits",
  "clip-path",
  "clip-rule",
  "color",
  "color-interpolation",
  "color-interpolation-filters",
  "color-profile",
  "color-rendering",
  "cx",
  "cy",
  "d",
  "dx",
  "dy",
  "diffuseconstant",
  "direction",
  "display",
  "divisor",
  "dominant-baseline",
  "dur",
  "edgemode",
  "elevation",
  "end",
  "fill",
  "fill-opacity",
  "fill-rule",
  "filter",
  "filterunits",
  "flood-color",
  "flood-opacity",
  "font-family",
  "font-size",
  "font-size-adjust",
  "font-stretch",
  "font-style",
  "font-variant",
  "font-weight",
  "fx",
  "fy",
  "g1",
  "g2",
  "glyph-name",
  "glyphref",
  "gradientunits",
  "gradienttransform",
  "height",
  "href",
  "id",
  "image-rendering",
  "in",
  "in2",
  "k",
  "k1",
  "k2",
  "k3",
  "k4",
  "kerning",
  "keypoints",
  "keysplines",
  "keytimes",
  "lang",
  "lengthadjust",
  "letter-spacing",
  "kernelmatrix",
  "kernelunitlength",
  "lighting-color",
  "local",
  "marker-end",
  "marker-mid",
  "marker-start",
  "markerheight",
  "markerunits",
  "markerwidth",
  "maskcontentunits",
  "maskunits",
  "max",
  "mask",
  "media",
  "method",
  "mode",
  "min",
  "name",
  "numoctaves",
  "offset",
  "operator",
  "opacity",
  "order",
  "orient",
  "orientation",
  "origin",
  "overflow",
  "paint-order",
  "path",
  "pathlength",
  "patterncontentunits",
  "patterntransform",
  "patternunits",
  "points",
  "preservealpha",
  "preserveaspectratio",
  "primitiveunits",
  "r",
  "rx",
  "ry",
  "radius",
  "refx",
  "refy",
  "repeatcount",
  "repeatdur",
  "restart",
  "result",
  "rotate",
  "scale",
  "seed",
  "shape-rendering",
  "specularconstant",
  "specularexponent",
  "spreadmethod",
  "startoffset",
  "stddeviation",
  "stitchtiles",
  "stop-color",
  "stop-opacity",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke",
  "stroke-width",
  "style",
  "surfacescale",
  "systemlanguage",
  "tabindex",
  "targetx",
  "targety",
  "transform",
  "transform-origin",
  "text-anchor",
  "text-decoration",
  "text-rendering",
  "textlength",
  "type",
  "u1",
  "u2",
  "unicode",
  "values",
  "viewbox",
  "visibility",
  "version",
  "vert-adv-y",
  "vert-origin-x",
  "vert-origin-y",
  "width",
  "word-spacing",
  "wrap",
  "writing-mode",
  "xchannelselector",
  "ychannelselector",
  "x",
  "x1",
  "x2",
  "xmlns",
  "y",
  "y1",
  "y2",
  "z",
  "zoomandpan"
], Ml = [
  "accent",
  "accentunder",
  "align",
  "bevelled",
  "close",
  "columnsalign",
  "columnlines",
  "columnspan",
  "denomalign",
  "depth",
  "dir",
  "display",
  "displaystyle",
  "encoding",
  "fence",
  "frame",
  "height",
  "href",
  "id",
  "largeop",
  "length",
  "linethickness",
  "lspace",
  "lquote",
  "mathbackground",
  "mathcolor",
  "mathsize",
  "mathvariant",
  "maxsize",
  "minsize",
  "movablelimits",
  "notation",
  "numalign",
  "open",
  "rowalign",
  "rowlines",
  "rowspacing",
  "rowspan",
  "rspace",
  "rquote",
  "scriptlevel",
  "scriptminsize",
  "scriptsizemultiplier",
  "selection",
  "separator",
  "separators",
  "stretchy",
  "subscriptshift",
  "supscriptshift",
  "symmetric",
  "voffset",
  "width",
  "xmlns"
];
Sl([
  Object.fromEntries(Al.map((e) => [e, ["*"]])),
  Object.fromEntries(Hl.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(Ml.map((e) => [e, ["math:*"]]))
]);
gr(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var Ol = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), Pl = /* @__PURE__ */ de("<option> </option>"), Nl = /* @__PURE__ */ de('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select></div>'), Il = /* @__PURE__ */ de('<!> <div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回视野</button></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function Ll(e, t) {
  _r(t, !0);
  const r = /* @__PURE__ */ bs(t, Ol), n = 2048, i = 35e-5, a = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], o = new rl(r);
  let u, c = null, f = null, d = null, b = null, m = /* @__PURE__ */ new Map(), T = /* @__PURE__ */ new Map(), M = Q(!1), S = Q(!1), y = Q("等待版图 mask"), v = Q(Xt({ enabled: !1 })), p = Q(Xt(me())), g = Q(""), x = Q(!1), E = Q(!1), P = Q("crosshair"), N = "", k = 0, D = { x: 0, y: 0, center_x: 0, center_y: 0 }, C = { angle: 0, rotation: 0 }, le = null, be = /* @__PURE__ */ new Set();
  function me() {
    return {
      transform_version: 2,
      revision: 0,
      center_x: 0,
      center_y: 0,
      pivot_x: 0,
      pivot_y: 0,
      scale: 1,
      rotation_deg: 0,
      preview_alpha: 0.35
    };
  }
  function K(s) {
    return JSON.parse(JSON.stringify(s || { enabled: !1 }));
  }
  function ve(s) {
    const h = Object.assign(Object.assign({}, me()), s || {});
    return Object.assign(Object.assign({}, h), {
      center_x: pe(h.center_x, 0),
      center_y: pe(h.center_y, 0),
      pivot_x: pe(h.pivot_x, 0),
      pivot_y: pe(h.pivot_y, 0),
      scale: we(pe(h.scale, 1), 0.01, 20),
      rotation_deg: Ge(pe(h.rotation_deg, 0)),
      preview_alpha: we(pe(h.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(pe(h.revision, 0)))
    });
  }
  function pe(s, h) {
    const _ = Number(s);
    return Number.isFinite(_) ? _ : h;
  }
  function ct(s) {
    return typeof s == "number" ? String(s) + "px" : s || "520px";
  }
  function we(s, h, _) {
    return Math.max(h, Math.min(_, s));
  }
  function Ge(s) {
    let h = ((s + 180) % 360 + 360) % 360 - 180;
    return h === -180 && (h = 180), h;
  }
  function Te(s) {
    return typeof s + ":" + String(s);
  }
  function ie() {
    var s;
    return V() ? Array.isArray((s = l(v).group_view) === null || s === void 0 ? void 0 : s.groups) ? l(v).group_view.groups : [] : [];
  }
  function V() {
    return l(v).transform_mode === "label_groups" && !!l(v).group_view;
  }
  function Se() {
    for (const s of ie())
      if (Te(s.group_id) === l(g)) return s;
    return null;
  }
  function fe() {
    const s = Se();
    return s ? String(s.label || "Label " + String(s.group_id)) : "";
  }
  function Z() {
    return Math.max(1, Number(l(v).target_width || c?.naturalWidth || 1));
  }
  function se() {
    return Math.max(1, Number(l(v).target_height || c?.naturalHeight || 1));
  }
  function Ke() {
    var s;
    const h = (s = m.get(l(g))) === null || s === void 0 ? void 0 : s.image;
    return Math.max(1, Number(l(v).source_width || f?.naturalWidth || h?.naturalWidth || 1));
  }
  function Xe() {
    var s;
    const h = (s = m.get(l(g))) === null || s === void 0 ? void 0 : s.image;
    return Math.max(1, Number(l(v).source_height || f?.naturalHeight || h?.naturalHeight || 1));
  }
  function ht() {
    return Math.min(1, n / Math.max(Z(), se()));
  }
  function Ue(s = l(g)) {
    var h;
    if (V()) {
      const w = (h = m.get(s)) === null || h === void 0 ? void 0 : h.view.foreground_bbox_xyxy;
      if (Array.isArray(w) && w.length >= 4) return w.slice(0, 4).map(Number);
    }
    const _ = l(v).foreground_bbox_xyxy;
    return Array.isArray(_) && _.length >= 4 ? _.slice(0, 4).map(Number) : [0, 0, Ke() - 1, Xe() - 1];
  }
  function $e(s, h, _) {
    if (!s) {
      h === k && _(null);
      return;
    }
    const w = new Image();
    w.onload = () => {
      h === k && _(w);
    }, w.onerror = () => {
      h === k && _(null);
    }, w.src = s;
  }
  function Ae(s, h) {
    const _ = Math.max(1, Number(l(v).source_width || s.naturalWidth || 1)), w = Math.max(1, Number(l(v).source_height || s.naturalHeight || 1)), O = document.createElement("canvas");
    O.width = _, O.height = w;
    const A = O.getContext("2d", { willReadFrequently: !0 });
    if (!A) return null;
    A.imageSmoothingEnabled = !1, A.drawImage(s, 0, 0, _, w);
    const B = A.getImageData(0, 0, _, w), R = document.createElement("canvas");
    R.width = _, R.height = w;
    const I = R.getContext("2d");
    if (!I) return null;
    const G = I.createImageData(_, w);
    for (let X = 0; X < B.data.length; X += 4) {
      const J = Math.max(B.data[X], B.data[X + 1], B.data[X + 2]);
      B.data[X + 3] > 0 && J >= 128 && (G.data[X] = h[0], G.data[X + 1] = h[1], G.data[X + 2] = h[2], G.data[X + 3] = 255);
    }
    return I.putImageData(G, 0, 0), { mask: O, tint: R };
  }
  function qe() {
    if (!f) {
      d = null, b = null;
      return;
    }
    const s = Ae(f, a[0]);
    d = s?.mask || null, b = s?.tint || null;
  }
  function Sr(s) {
    if (!s.image) {
      s.maskCanvas = null, s.tintCanvas = null, s.ready = !1;
      return;
    }
    const h = Ae(s.image, s.color);
    s.maskCanvas = h?.mask || null, s.tintCanvas = h?.tint || null, s.ready = !!h;
  }
  function et() {
    le && (clearTimeout(le), le = null);
  }
  function Ar(s) {
    const h = l(v).group_view, _ = Array.isArray(h?.groups) ? h.groups : [], w = l(v).group_intent, O = !!h && !!w && String(w.selection_signature || "") === String(h.selection_signature || ""), A = /* @__PURE__ */ new Map();
    if (O && Array.isArray(w?.transforms))
      for (const J of w.transforms)
        !J || J.group_id === void 0 || !J.transform || A.set(Te(J.group_id), ve(J.transform));
    const B = ve(l(v).transform), R = /* @__PURE__ */ new Map(), I = /* @__PURE__ */ new Map(), G = /* @__PURE__ */ new Set();
    for (let J = 0; J < _.length; J++) {
      const Pe = _[J];
      if (!Pe || Pe.group_id === void 0 || Pe.group_id === null) continue;
      const je = Te(Pe.group_id);
      if (G.has(je)) continue;
      G.add(je);
      const Qe = {
        view: Pe,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: a[J % a.length]
      };
      R.set(je, Qe), I.set(je, ve(A.get(je) || B));
    }
    m = R, T = I, be = /* @__PURE__ */ new Set();
    const X = O && w?.active_group_id !== void 0 && w.active_group_id !== null ? Te(w.active_group_id) : "";
    H(
      g,
      R.has(X) ? X : R.keys().next().value || "",
      !0
    ), H(p, ve(T.get(l(g)) || B), !0), l(g) && T.set(l(g), l(p)), f = null, d = null, b = null, H(S, !1);
    for (const [J, Pe] of R)
      $e(Pe.view.mask_image, s, (je) => {
        const Qe = m.get(J);
        !Qe || Qe !== Pe || (Qe.image = je, Sr(Qe), _e());
      });
    H(
      y,
      l(v).status || (_.length > 0 ? "已加载 " + String(_.length) + " 个 Label；当前：" + fe() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function Hr(s) {
    et(), k += 1;
    const h = k;
    H(v, K(s), !0), H(M, !1), m = /* @__PURE__ */ new Map(), T = /* @__PURE__ */ new Map(), H(g, ""), H(x, !1), H(E, !1), $e(l(v).base_image, h, (_) => {
      c = _, H(M, !!_), _e();
    }), V() ? Ar(h) : (H(p, ve(l(v).transform), !0), H(y, l(v).status || "编辑器已加载", !0), H(S, !1), $e(l(v).mask_image, h, (_) => {
      f = _, H(S, !!_), qe(), _e();
    }));
  }
  Le(() => {
    const s = JSON.stringify(o.props.value || null);
    s !== N && (N = s, Hr(o.props.value));
  }), Ja(() => {
    k += 1, et();
  });
  function kt(s) {
    const h = Number(s.rotation_deg || 0) * Math.PI / 180, _ = Number(s.scale || 1), w = Math.cos(h), O = Math.sin(h), A = _ * w, B = _ * O, R = Number(s.center_x || 0) - A * Number(s.pivot_x || 0) + B * Number(s.pivot_y || 0), I = Number(s.center_y || 0) - B * Number(s.pivot_x || 0) - A * Number(s.pivot_y || 0);
    return [A, B, -B, A, R, I];
  }
  function dt(s, h, _ = l(p)) {
    const [w, O, A, B, R, I] = kt(_);
    return { x: w * s + A * h + R, y: O * s + B * h + I };
  }
  function Y(s, h, _ = l(p)) {
    const [w, O, A, B, R, I] = kt(_), G = w * B - O * A;
    if (Math.abs(G) < 1e-9) return { x: -1, y: -1 };
    const X = s - R, J = h - I;
    return { x: (B * X - A * J) / G, y: (-O * X + w * J) / G };
  }
  function ee(s) {
    const h = u.getBoundingClientRect();
    return {
      x: (s.clientX - h.left) / Math.max(1, h.width) * Z(),
      y: (s.clientY - h.top) / Math.max(1, h.height) * se()
    };
  }
  function ce(s, h, _) {
    if (!s) return !1;
    const w = Math.round(h), O = Math.round(_);
    if (w < 0 || O < 0 || w >= s.width || O >= s.height) return !1;
    const A = s.getContext("2d", { willReadFrequently: !0 });
    if (!A) return !1;
    const B = A.getImageData(w, O, 1, 1).data;
    return B[3] > 0 && Math.max(B[0], B[1], B[2]) >= 128;
  }
  function oe(s) {
    return !V() || s === l(g) ? l(p) : T.get(s) || l(p);
  }
  function ue(s) {
    H(p, ve(s), !0), V() && l(g) && T.set(l(g), l(p));
  }
  function Me(s, h) {
    const _ = Y(s, h, l(p));
    return ce(d, _.x, _.y);
  }
  function We(s, h, _) {
    const w = m.get(s);
    if (!w?.ready) return !1;
    const O = Y(h, _, oe(s));
    return ce(w.maskCanvas, O.x, O.y);
  }
  function Ce(s, h) {
    if (l(g) && We(l(g), s, h)) return l(g);
    const _ = ie().map((w) => Te(w.group_id)).reverse();
    for (const w of _)
      if (w !== l(g) && We(w, s, h)) return w;
    return "";
  }
  function Fe(s = l(g), h = oe(s)) {
    const [_, w, O, A] = Ue(s);
    return [
      dt(_, w, h),
      dt(O, w, h),
      dt(O, A, h),
      dt(_, A, h)
    ];
  }
  function vt() {
    const s = Fe();
    let h = s[0];
    for (const A of s)
      (A.y < h.y || Math.abs(A.y - h.y) < 1e-6 && A.x > h.x) && (h = A);
    const _ = Number(l(p).rotation_deg || 0) * Math.PI / 180, w = Math.cos(_ - Math.PI / 4), O = Math.sin(_ - Math.PI / 4);
    return { x: h.x + w * 34, y: h.y + O * 34 };
  }
  function Jt() {
    const s = u?.getBoundingClientRect();
    return s ? Math.max(8, 14 * Z() / Math.max(1, s.width)) : 14;
  }
  function Qt(s, h) {
    if (V() && !l(g)) return !1;
    const _ = vt(), w = Jt();
    return Math.hypot(s - _.x, h - _.y) <= w;
  }
  function Dt(s) {
    const h = V() ? l(g) : "";
    ue(Object.assign(Object.assign({}, l(p)), {
      revision: Number(l(p).revision || 0) + 1,
      origin: s,
      scale: we(Number(l(p).scale || 1), 0.01, 20),
      rotation_deg: Ge(Number(l(p).rotation_deg || 0))
    })), h && be.add(h);
  }
  function Mr() {
    const s = [];
    for (const h of ie()) {
      const _ = Te(h.group_id);
      s.push({
        group_id: h.group_id,
        transform: Object.assign({}, ve(oe(_)))
      });
    }
    return s;
  }
  function Kt() {
    const s = {
      enabled: l(v).enabled,
      transform: Object.assign({}, ve(l(p))),
      target_width: l(v).target_width,
      target_height: l(v).target_height
    };
    V() && (s.transform_mode = "label_groups", s.group_intent = l(v).group_intent ? JSON.parse(JSON.stringify(l(v).group_intent)) : null), o.props.value = s, N = JSON.stringify(s);
  }
  function $t(s, h, _, w = !0) {
    var O, A;
    if (!V()) return;
    _ && Dt(s);
    const B = l(v).group_view, R = Se(), I = ie().filter((X) => be.has(Te(X.group_id))).map((X) => X.group_id), G = Number(((O = l(v).group_intent) === null || O === void 0 ? void 0 : O.transform_set_revision) || 0);
    H(
      v,
      Object.assign(Object.assign({}, l(v)), {
        group_intent: {
          selection_signature: String(B?.selection_signature || ""),
          transform_set_revision: G + 1,
          active_group_id: (A = R?.group_id) !== null && A !== void 0 ? A : null,
          changed_group_ids: I,
          transforms: Mr()
        }
      }),
      !0
    ), H(y, h, !0), Kt(), w && (et(), o.dispatch("change")), _e();
  }
  function tt(s, h, _ = !0) {
    if (V()) {
      $t(s, h, !0, _);
      return;
    }
    Dt(s), H(
      v,
      Object.assign(Object.assign({}, l(v)), {
        enabled: !0,
        transform: Object.assign({}, l(p)),
        status: h
      }),
      !0
    ), H(y, h, !0), Kt(), _ && (et(), o.dispatch("change")), _e();
  }
  function q(s, h, _ = 140) {
    tt(s, h, !1), et(), le = setTimeout(
      () => {
        le = null, o.dispatch("change");
      },
      _
    );
  }
  function te(s, h, _) {
    if (c && l(M)) {
      s.imageSmoothingEnabled = !0, s.drawImage(c, 0, 0, h, _);
      return;
    }
    s.fillStyle = "#f8fafc", s.fillRect(0, 0, h, _), s.fillStyle = "#64748b", s.font = "18px sans-serif", s.fillText("请先上传图像", 24, 42);
  }
  function He(s, h, _, w) {
    var O;
    s.save(), s.globalAlpha = we(Number((O = w.preview_alpha) !== null && O !== void 0 ? O : 0.35), 0, 1);
    const [A, B, R, I, G, X] = kt(w);
    s.setTransform(h * A, h * B, h * R, h * I, h * G, h * X), s.imageSmoothingEnabled = !1, s.drawImage(_, 0, 0, _.width, _.height), s.restore();
  }
  function rt(s, h) {
    const _ = Fe();
    s.save(), s.lineJoin = "round", s.lineWidth = Math.max(2.5, h / 700), s.strokeStyle = "rgba(0,0,0,0.82)", s.beginPath(), s.moveTo(_[0].x, _[0].y);
    for (let A = 1; A < _.length; A++) s.lineTo(_[A].x, _[A].y);
    s.closePath(), s.stroke(), s.lineWidth = Math.max(1.8, h / 1e3), s.strokeStyle = "#00ff66", s.stroke();
    const w = vt(), O = _.reduce((A, B) => B.y < A.y || Math.abs(B.y - A.y) < 1e-6 && B.x > A.x ? B : A, _[0]);
    s.strokeStyle = "#0f172a", s.lineWidth = Math.max(2, h / 900), s.beginPath(), s.moveTo(O.x, O.y), s.lineTo(w.x, w.y), s.stroke(), s.fillStyle = l(E) ? "#ffb000" : "#ffffff", s.strokeStyle = "#00ff66", s.lineWidth = Math.max(2, h / 900), s.beginPath(), s.arc(w.x, w.y, Jt(), 0, Math.PI * 2), s.fill(), s.stroke(), s.restore();
  }
  function Oe(s, h, _) {
    const w = m.get(h);
    if (!w) return;
    const O = Fe(h, oe(h)), A = Math.min(...O.map((J) => J.x)), B = Math.min(...O.map((J) => J.y)), R = String(w.view.label || "Label " + String(w.view.group_id));
    s.save(), s.font = "600 13px sans-serif";
    const I = s.measureText(R).width + 12, G = we(A, 2, Math.max(2, Z() - I - 2)), X = we(B - 23, 2, Math.max(2, se() - 22));
    s.fillStyle = _ ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.76)", s.fillRect(G, X, I, 20), s.fillStyle = "#ffffff", s.fillText(R, G + 6, X + 14), s.restore();
  }
  function _e() {
    var s, h;
    if (!u) return;
    const _ = Z(), w = se(), O = ht();
    u.width = Math.max(1, Math.round(_ * O)), u.height = Math.max(1, Math.round(w * O));
    const A = u.getContext("2d");
    if (A && (A.setTransform(O, 0, 0, O, 0, 0), A.clearRect(0, 0, _, w), te(A, _, w), l(v).enabled !== !1)) {
      if (V()) {
        const R = ie().map((I) => Te(I.group_id)).filter((I) => I !== l(g));
        l(g) && R.push(l(g));
        for (const I of R) {
          const G = m.get(I);
          G?.ready && G.tintCanvas && He(A, O, G.tintCanvas, oe(I));
        }
        A.setTransform(O, 0, 0, O, 0, 0);
        for (const I of R)
          !((s = m.get(I)) === null || s === void 0) && s.ready && Oe(A, I, I === l(g));
        l(g) && (!((h = m.get(l(g))) === null || h === void 0) && h.ready) && rt(A, _);
        return;
      }
      b && l(S) && (He(A, O, b, l(p)), A.setTransform(O, 0, 0, O, 0, 0), rt(A, _));
    }
  }
  function ye() {
    var s;
    return l(v).enabled ? V() ? !!l(g) && !!(!((s = m.get(l(g))) === null || s === void 0) && s.ready) : l(S) && !!d : !1;
  }
  function nt(s) {
    return !V() || !m.has(s) || s === l(g) ? !1 : (l(g) && T.set(l(g), ve(l(p))), H(g, s, !0), H(p, ve(T.get(s)), !0), T.set(s, l(p)), H(y, "当前 Label：" + fe()), _e(), !0);
  }
  function Ze(s) {
    const h = s.currentTarget.value;
    nt(h) && $t("select_group", "已选择 Label：" + fe(), !1, !0);
  }
  function pt(s) {
    if (s.button !== 0) return;
    if (!ye()) {
      H(y, "请先启用并加载版图 mask"), _e();
      return;
    }
    et();
    const h = ee(s);
    if (Qt(h.x, h.y)) {
      H(P, "grabbing"), H(E, !0), C = {
        angle: Math.atan2(h.y - l(p).center_y, h.x - l(p).center_x) * 180 / Math.PI,
        rotation: Number(l(p).rotation_deg || 0)
      }, u.setPointerCapture(s.pointerId);
      return;
    }
    if (V()) {
      const _ = Ce(h.x, h.y);
      if (!_) {
        H(y, "请点中任一 Label mask 前景后拖动"), _e();
        return;
      }
      nt(_);
    } else if (!Me(h.x, h.y)) {
      H(y, "请点中版图 mask 前景后拖动"), _e();
      return;
    }
    H(P, "grabbing"), H(x, !0), D = {
      x: h.x,
      y: h.y,
      center_x: Number(l(p).center_x || 0),
      center_y: Number(l(p).center_y || 0)
    }, u.setPointerCapture(s.pointerId);
  }
  function mt(s) {
    if (!ye()) {
      H(P, "not-allowed");
      return;
    }
    const h = V() ? !!Ce(s.x, s.y) : Me(s.x, s.y);
    if (Qt(s.x, s.y) || h) {
      H(P, "grab");
      return;
    }
    H(P, "crosshair");
  }
  function Gt(s) {
    const h = ee(s);
    if (!l(x) && !l(E)) {
      mt(h);
      return;
    }
    if (H(P, "grabbing"), l(x))
      ue(Object.assign(Object.assign({}, l(p)), {
        center_x: D.center_x + h.x - D.x,
        center_y: D.center_y + h.y - D.y
      })), H(y, "正在拖动 " + (V() ? fe() : "版图") + "；松开后同步变换");
    else if (l(E)) {
      const _ = Math.atan2(h.y - l(p).center_y, h.x - l(p).center_x) * 180 / Math.PI;
      ue(Object.assign(Object.assign({}, l(p)), {
        rotation_deg: Ge(C.rotation + _ - C.angle)
      })), H(y, "正在旋转 " + (V() ? fe() : "版图") + "；松开后同步变换");
    }
    _e();
  }
  function it() {
    !l(x) && !l(E) && H(P, "crosshair");
  }
  function Ye(s) {
    if (l(x) || l(E)) {
      H(x, !1), H(E, !1);
      try {
        u.releasePointerCapture(s.pointerId);
      } catch {
      }
      mt(ee(s));
      const h = V() ? fe() : "Canvas";
      tt("canvas", h + " 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function Re(s) {
    if (!ye()) return;
    s.preventDefault();
    const h = ee(s);
    if (V()) {
      const R = Ce(h.x, h.y);
      R && nt(R);
    }
    const _ = Y(h.x, h.y, l(p)), w = Math.exp(-s.deltaY * i), O = we(Number(l(p).scale || 1) * w, 0.01, 20);
    let A = Object.assign(Object.assign({}, l(p)), { scale: O });
    const B = dt(_.x, _.y, A);
    A = Object.assign(Object.assign({}, A), {
      center_x: Number(A.center_x || 0) + h.x - B.x,
      center_y: Number(A.center_y || 0) + h.y - B.y
    }), ue(A), q("canvas", "滚轮缩放已同步: scale=" + O.toFixed(3));
  }
  function Je() {
    ye() && (ue(Object.assign(Object.assign({}, l(p)), {
      center_x: Z() / 2,
      center_y: se() / 2,
      scale: 1,
      rotation_deg: 0
    })), tt("reset", "居中归一 " + (V() ? fe() : "版图") + "：scale=1，rotation=0"));
  }
  function gt() {
    ye() && (ue(Object.assign(Object.assign({}, l(p)), { center_x: Z() / 2, center_y: se() / 2 })), tt("center", "居中 " + (V() ? fe() : "版图") + "：保留缩放和旋转"));
  }
  function bt() {
    if (!ye()) return;
    const [s, h, _, w] = Ue(), O = Math.max(1, _ - s + 1), A = Math.max(1, w - h + 1), B = we(Math.min(Z() / O, se() / A) * 0.9, 0.01, 20);
    ue(Object.assign(Object.assign({}, l(p)), {
      center_x: Z() / 2,
      center_y: se() / 2,
      scale: B
    })), tt("fit", "适配 " + (V() ? fe() : "版图") + "：scale=" + B.toFixed(3));
  }
  function er() {
    if (!ye()) return;
    const s = Fe(), h = Math.min(...s.map((I) => I.x)), _ = Math.max(...s.map((I) => I.x)), w = Math.min(...s.map((I) => I.y)), O = Math.max(...s.map((I) => I.y));
    let A = 0, B = 0;
    const R = Math.max(20, Z() * 0.03);
    if (_ < R ? A = R - _ : h > Z() - R && (A = Z() - R - h), O < R ? B = R - O : w > se() - R && (B = se() - R - w), A === 0 && B === 0) {
      H(y, (V() ? fe() : "版图") + " 已经在视野内");
      return;
    }
    ue(Object.assign(Object.assign({}, l(p)), {
      center_x: Number(l(p).center_x || 0) + A,
      center_y: Number(l(p).center_y || 0) + B
    })), tt("bring_into_view", "已找回 " + (V() ? fe() : "版图") + " 到视野内");
  }
  {
    let s = Be(() => l(x) || l(E) ? "focus" : "base");
    al(e, {
      get visible() {
        return o.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return l(s);
      },
      padding: !1,
      get elem_id() {
        return o.shared.elem_id;
      },
      get elem_classes() {
        return o.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return o.shared.container;
      },
      get scale() {
        return o.shared.scale;
      },
      get min_width() {
        return o.shared.min_width;
      },
      children: (h, _) => {
        var w = Il(), O = xe(w);
        Tl(O, ys(
          {
            get autoscroll() {
              return o.shared.autoscroll;
            },
            get i18n() {
              return o.i18n;
            }
          },
          () => o.shared.loading_status,
          {
            on_clear_status: () => o.dispatch("clear_status", o.shared.loading_status)
          }
        ));
        var A = W(O, 2), B = ae(A);
        {
          var R = (_t) => {
            var tr = Nl(), st = W(ae(tr), 2);
            Jr(st, 21, ie, Zr, (Or, Pr) => {
              var nr = Pl(), Wi = ae(nr), yn = {};
              ne(
                (Zi, xn) => {
                  ge(Wi, Zi), yn !== (yn = xn) && (nr.value = (nr.__value = xn) ?? "");
                },
                [
                  () => l(Pr).label || "Label " + String(l(Pr).group_id),
                  () => Te(l(Pr).group_id)
                ]
              ), U(Or, nr);
            });
            var rr;
            gi(st), ne(() => {
              rr !== (rr = l(g)) && (st.value = (st.__value = l(g)) ?? "", hr(st, l(g)));
            }), Ie("change", st, Ze), U(_t, tr);
          }, I = Be(() => V());
          re(B, (_t) => {
            l(I) && _t(R);
          });
        }
        var G = W(B, 2), X = ae(G), J = W(X, 2), Pe = W(J, 2), je = W(Pe, 2), Qe = W(G, 2), at = ae(Qe);
        gn(at, (_t) => u = _t, () => u);
        var Xi = W(Qe, 2), qi = ae(Xi);
        ne(
          (_t, tr, st, rr, Or) => {
            De(A, _t), X.disabled = tr, J.disabled = st, Pe.disabled = rr, je.disabled = Or, De(at, "cursor:" + l(P)), ge(qi, l(y));
          },
          [
            () => "min-height:" + ct(o.props.height),
            () => !ye(),
            () => !ye(),
            () => !ye(),
            () => !ye()
          ]
        ), Ie("click", X, Je), Ie("click", J, gt), Ie("click", Pe, bt), Ie("click", je, er), Ie("pointerdown", at, pt), Ie("pointermove", at, Gt), Ie("pointerup", at, Ye), Ie("pointercancel", at, Ye), Ie("pointerleave", at, it), Ie("wheel", at, Re), U(h, w);
      },
      $$slots: { default: !0 }
    });
  }
  br();
}
export {
  Ll as default
};
