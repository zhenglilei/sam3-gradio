import { i as Dr, g as Nn, o as Mi, n as Ze, u as ie, s as Pi, r as xr, m as $e, a as A, b as f, t as Ur, d as Oi, q as Bi, c as Ln, e as rt, f as qt, h as jt, j as Ii, T as Ni, k as Li, l as Vt, p as et, v as Fr, w as nt, x as Cn, y as Rn, z as kn, A as Pt, E as Wt, B as ft, C as Dn, D as we, F as $r, G as Ci, H as Un, I as Gr, J as Ri, K as en, L as ki, M as Di, N as Ge, O as Fn, P as lr, Q as Ui, R as Fi, S as Gi, U as ji, V as Gn, W as jr, X as tn, Y as rn, Z as Vi, _ as zi, $ as Xi, a0 as qi, a1 as Wi, a2 as Zi, a3 as Yi, a4 as Ji, a5 as Vr, a6 as Qi, a7 as jn, a8 as Zt, a9 as Ki, aa as $i, ab as ea, ac as ta, ad as ra, ae as na, af as zr, ag as ia, ah as aa, ai as Ee, aj as Er, ak as wr, al as sa, am as oa, an as Ht, ao as la, ap as ua, aq as fa, ar as ca, as as ha, at as Vn, au as Et, av as W, aw as da, ax as nn, ay as ma, az as de, aA as Yt, aB as Jt, aC as V, aD as He, aE as $, aF as pa, aG as re, aH as he, aI as Me, aJ as va } from "./render-BGdAxg9I.js";
function zn(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const ga = [];
function ba(e, t = !1, r = !1) {
  return Ut(e, /* @__PURE__ */ new Map(), "", ga, null, r);
}
function Ut(e, t, r, n, i = null, a = !1) {
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
    if (Dr(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var u = 0; u < e.length; u += 1) {
        var l = e[u];
        u in e && (s[u] = Ut(l, t, r, n, null, a));
      }
      return s;
    }
    if (Nn(e) === Mi) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var c of Object.keys(e))
        s[c] = Ut(
          // @ts-expect-error
          e[c],
          t,
          r,
          n,
          null,
          a
        );
      return s;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !a)
      return Ut(
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
function Xr(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), Ze;
  const n = ie(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const ot = [];
function _a(e, t) {
  return {
    subscribe: Ot(e, t).subscribe
  };
}
function Ot(e, t = Ze) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Pi(e, s) && (e = s, r)) {
      const u = !ot.length;
      for (const l of n)
        l[1](), ot.push(l, e);
      if (u) {
        for (let l = 0; l < ot.length; l += 2)
          ot[l][0](ot[l + 1]);
        ot.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = Ze) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || Ze), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function vt(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return _a(r, (o, s) => {
    let u = !1;
    const l = [];
    let c = 0, v = Ze;
    const g = () => {
      if (c)
        return;
      v();
      const d = t(n ? l[0] : l, o, s);
      a ? o(d) : v = typeof d == "function" ? d : Ze;
    }, w = i.map(
      (d, m) => Xr(
        d,
        (P) => {
          l[m] = P, c &= ~(1 << m), u && g();
        },
        () => {
          c |= 1 << m;
        }
      )
    );
    return u = !0, g(), function() {
      xr(w), v(), u = !1;
    };
  });
}
function ya(e) {
  let t;
  return Xr(e, (r) => t = r)(), t;
}
let Rt = !1, Tr = /* @__PURE__ */ Symbol("unmounted");
function an(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: $e(void 0),
    unsubscribe: Ze
  };
  if (n.store !== e && !(Tr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = Ze;
    else {
      var i = !0;
      n.unsubscribe = Xr(e, (a) => {
        i ? n.source.v = a : A(n.source, a);
      }), i = !1;
    }
  return e && Tr in r ? ya(e) : f(n.source);
}
function xa() {
  const e = {};
  function t() {
    Ur(() => {
      for (var r in e)
        e[r].unsubscribe();
      Oi(e, Tr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = Rt;
  try {
    return Rt = !1, [e(), Rt];
  } finally {
    Rt = t;
  }
}
function wa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Bi(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Ta = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Sa(e) {
  return (
    /** @type {string} */
    Ta?.createHTML(e) ?? e
  );
}
function Xn(e) {
  var t = Ln("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function ht(e, t) {
  var r = (
    /** @type {Effect} */
    qt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function ue(e, t) {
  var r = (t & Ni) !== 0, n = (t & Li) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Xn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    jt(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ii ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        jt(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      ht(s, u);
    } else
      ht(o, o);
    return o;
  };
}
// @__NO_SIDE_EFFECTS__
function Aa(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var o = (
        /** @type {DocumentFragment} */
        Xn(i)
      ), s = (
        /** @type {Element} */
        jt(o)
      );
      a = /** @type {Element} */
      jt(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return ht(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function qn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function Re(e = "") {
  {
    var t = rt(e + "");
    return ht(t, t), t;
  }
}
function ut() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = rt();
  return e.append(t, r), ht(t, r), e;
}
function R(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Qt {
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
        Vt(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Vt(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (et(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Rn(o, l), l.append(rt()), this.#e.set(a, { effect: o, fragment: l });
          } else
            et(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Fr(o, s, !1)) : s();
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
      r.includes(n) || (et(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Cn
    ), i = kn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = rt();
        a.append(o), this.#e.set(t, {
          effect: nt(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          nt(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [s, u] of this.#r)
        s === t ? n.unskip_effect(u) : n.skip_effect(u);
      for (const [s, u] of this.#e)
        s === t ? n.unskip_effect(u.effect) : n.skip_effect(u.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Ha(e, t, ...r) {
  var n = new Qt(e);
  Pt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Wt);
}
function Ma(e) {
  ft === null && zn(), Dn && ft.l !== null ? Oa(ft).m.push(e) : we(() => {
    const t = ie(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Pa(e) {
  ft === null && zn(), Ma(() => () => ie(e));
}
function Oa(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function Q(e, t, r = !1) {
  var n = new Qt(e), i = r ? Wt : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  Pt(() => {
    var o = !1;
    t((s, u = 0) => {
      o = !0, a(u, s);
    }), o || a(-1, null);
  }, i);
}
function sn(e, t) {
  return t;
}
function Ba(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let v = t[s];
    Fr(
      v,
      () => {
        if (a) {
          if (a.pending.delete(v), a.done.add(v), a.pending.size === 0) {
            var g = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Sr(e, Gr(a.done)), g.delete(a), g.size === 0 && (e.outrogroups = null);
          }
        } else
          o -= 1;
      },
      !1
    );
  }
  if (o === 0) {
    var u = n.length === 0 && r !== null;
    if (u) {
      var l = (
        /** @type {Element} */
        r
      ), c = (
        /** @type {Element} */
        l.parentNode
      );
      Fi(c), c.append(l), e.items.clear();
    }
    Sr(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Sr(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const o of e.pending.values())
      for (const s of o)
        n.add(
          /** @type {EachItem} */
          e.items.get(s).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var a = t[i];
    if (n?.has(a)) {
      a.f |= Ge;
      const o = document.createDocumentFragment();
      Rn(a, o);
    } else
      et(t[i], r);
  }
}
var on;
function ln(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = Un(() => {
    var h = r();
    return (
      /** @type {V[]} */
      Dr(h) ? h : h == null ? [] : Gr(h)
    );
  }), c, v = /* @__PURE__ */ new Map(), g = !0;
  function w(h) {
    (P.effect.f & Fn) === 0 && (P.pending.delete(h), P.fallback = u, Ia(P, c, o, t, n), u !== null && (c.length === 0 ? (u.f & Ge) === 0 ? Vt(u) : (u.f ^= Ge, St(u, null, o)) : Fr(u, () => {
      u = null;
    })));
  }
  function d(h) {
    P.pending.delete(h);
  }
  var m = Pt(() => {
    c = /** @type {V[]} */
    f(l);
    for (var h = c.length, b = /* @__PURE__ */ new Set(), T = (
      /** @type {Batch} */
      Cn
    ), y = kn(), x = 0; x < h; x += 1) {
      var H = c[x], O = n(H, x), B = g ? null : s.get(O);
      B ? (B.v && $r(B.v, H), B.i && $r(B.i, x), y && T.unskip_effect(B.e)) : (B = Na(
        s,
        g ? o : on ??= rt(),
        H,
        O,
        x,
        i,
        t,
        r
      ), g || (B.e.f |= Ge), s.set(O, B)), b.add(O);
    }
    if (h === 0 && a && !u && (g ? u = nt(() => a(o)) : (u = nt(() => a(on ??= rt())), u.f |= Ge)), h > b.size && Ci(), !g)
      if (v.set(T, b), y) {
        for (const [G, U] of s)
          b.has(G) || T.skip_effect(U.e);
        T.oncommit(w), T.ondiscard(d);
      } else
        w(T);
    f(l);
  }), P = { effect: m, items: s, pending: v, outrogroups: null, fallback: u };
  g = !1;
}
function wt(e) {
  for (; e !== null && (e.f & Ui) === 0; )
    e = e.next;
  return e;
}
function Ia(e, t, r, n, i) {
  var a = t.length, o = e.items, s = wt(e.effect.first), u, l = null, c = [], v = [], g, w, d, m;
  for (m = 0; m < a; m += 1) {
    if (g = t[m], w = i(g, m), d = /** @type {EachItem} */
    o.get(w).e, e.outrogroups !== null)
      for (const B of e.outrogroups)
        B.pending.delete(d), B.done.delete(d);
    if ((d.f & lr) !== 0 && Vt(d), (d.f & Ge) !== 0)
      if (d.f ^= Ge, d === s)
        St(d, null, r);
      else {
        var P = l ? l.next : s;
        d === e.effect.last && (e.effect.last = d.prev), d.prev && (d.prev.next = d.next), d.next && (d.next.prev = d.prev), qe(e, l, d), qe(e, d, P), St(d, P, r), l = d, c = [], v = [], s = wt(l.next);
        continue;
      }
    if (d !== s) {
      if (u !== void 0 && u.has(d)) {
        if (c.length < v.length) {
          var h = v[0], b;
          l = h.prev;
          var T = c[0], y = c[c.length - 1];
          for (b = 0; b < c.length; b += 1)
            St(c[b], h, r);
          for (b = 0; b < v.length; b += 1)
            u.delete(v[b]);
          qe(e, T.prev, y.next), qe(e, l, T), qe(e, y, h), s = h, l = y, m -= 1, c = [], v = [];
        } else
          u.delete(d), St(d, s, r), qe(e, d.prev, d.next), qe(e, d, l === null ? e.effect.first : l.next), qe(e, l, d), l = d;
        continue;
      }
      for (c = [], v = []; s !== null && s !== d; )
        (u ??= /* @__PURE__ */ new Set()).add(s), v.push(s), s = wt(s.next);
      if (s === null)
        continue;
    }
    (d.f & Ge) === 0 && c.push(d), l = d, s = wt(d.next);
  }
  if (e.outrogroups !== null) {
    for (const B of e.outrogroups)
      B.pending.size === 0 && (Sr(e, Gr(B.done)), e.outrogroups?.delete(B));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var x = [];
    if (u !== void 0)
      for (d of u)
        (d.f & lr) === 0 && x.push(d);
    for (; s !== null; )
      (s.f & lr) === 0 && s !== e.fallback && x.push(s), s = wt(s.next);
    var H = x.length;
    if (H > 0) {
      var O = null;
      Ba(e, x, O);
    }
  }
}
function Na(e, t, r, n, i, a, o, s) {
  var u = (o & ki) !== 0 ? (o & Di) === 0 ? $e(r, !1, !1) : en(r) : null, l = (o & Ri) !== 0 ? en(i) : null;
  return {
    v: u,
    i: l,
    e: nt(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function St(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Ge) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        Gi(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function qe(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Ar(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function La(e, t, r) {
  var n = new Qt(e);
  Pt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Wt);
}
const Ca = () => performance.now(), Pe = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Ca(),
  tasks: /* @__PURE__ */ new Set()
};
function Wn() {
  const e = Pe.now();
  Pe.tasks.forEach((t) => {
    t.c(e) || (Pe.tasks.delete(t), t.f());
  }), Pe.tasks.size !== 0 && Pe.tick(Wn);
}
function Ra(e) {
  let t;
  return Pe.tasks.size === 0 && Pe.tick(Wn), {
    promise: new Promise((r) => {
      Pe.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Pe.tasks.delete(t);
    }
  };
}
function ka(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new Qt(s, !1);
  Pt(() => {
    const l = t() || null;
    var c = l === "svg" ? ji : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (v) => {
      if (l) {
        if (o = Ln(l, c), ht(o, o), n) {
          var g = null, w = o.appendChild(rt());
          n(o, w), g?.remove();
        }
        qt.nodes.end = o, v.before(o);
      }
    }), () => {
    };
  }, Wt), Ur(() => {
  });
}
function Da(e, t) {
  var r = void 0, n;
  Gn(() => {
    r !== (r = t()) && (n && (et(n), n = null), r && (n = nt(() => {
      jr(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Zn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = Zn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ua() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = Zn(e)) && (n && (n += " "), n += t);
  return n;
}
function Fa(e) {
  return typeof e == "object" ? Ua(e) : e ?? "";
}
const un = [...` 	
\r\f \v\uFEFF`];
function Ga(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || un.includes(n[o - 1])) && (s === n.length || un.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function fn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function ur(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function ja(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(ur)), i && u.push(...Object.keys(i).map(ur));
      var l = 0, c = -1;
      const m = e.length;
      for (var v = 0; v < m; v++) {
        var g = e[v];
        if (s ? g === "/" && e[v - 1] === "*" && (s = !1) : a ? a === g && (a = !1) : g === "/" && e[v + 1] === "*" ? s = !0 : g === '"' || g === "'" ? a = g : g === "(" ? o++ : g === ")" && o--, !s && a === !1 && o === 0) {
          if (g === ":" && c === -1)
            c = v;
          else if (g === ";" || v === m - 1) {
            if (c !== -1) {
              var w = ur(e.substring(l, c).trim());
              if (!u.includes(w)) {
                g !== ";" && v++;
                var d = e.substring(l, v).trim();
                r += " " + d + ";";
              }
            }
            l = v + 1, c = -1;
          }
        }
      }
    }
    return n && (r += fn(n)), i && (r += fn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function tt(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[tn]
  );
  if (o !== r || o === void 0) {
    var s = Ga(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[tn] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function fr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Oe(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[rn]
  );
  if (i !== t) {
    var a = ja(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[rn] = t;
  } else n && (Array.isArray(n) ? (fr(e, r?.[0], n[0]), fr(e, r?.[1], n[1], "important")) : fr(e, r, n));
  return n;
}
function Hr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Dr(t))
      return Vi();
    for (var n of e.options)
      n.selected = t.includes(cn(n));
    return;
  }
  for (n of e.options) {
    var i = cn(n);
    if (zi(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Va(e) {
  var t = new MutationObserver(() => {
    Hr(e, e.__value);
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
  }), Ur(() => {
    t.disconnect();
  });
}
function cn(e) {
  return "__value" in e ? e.__value : e.value;
}
const At = /* @__PURE__ */ Symbol("class"), lt = /* @__PURE__ */ Symbol("style"), Yn = /* @__PURE__ */ Symbol("is custom element"), Jn = /* @__PURE__ */ Symbol("is html"), za = Vr ? "input" : "INPUT", Xa = Vr ? "option" : "OPTION", qa = Vr ? "select" : "SELECT";
function Wa(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function ct(e, t, r, n) {
  var i = Qn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Xi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Kn(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Za(e, t, r, n, i = !1, a = !1) {
  var o = Qn(e), s = o[Yn], u = !o[Jn], l = t || {}, c = e.nodeName === Xa;
  for (var v in t)
    v in r || (r[v] = null);
  r.class ? r.class = Fa(r.class) : r.class = null, r[lt] && (r.style ??= null);
  var g = Kn(e);
  if (e.nodeName === za && "type" in r && ("value" in r || "__value" in r)) {
    var w = r.type;
    (w !== l.type || w === void 0 && e.hasAttribute("type")) && (l.type = w, ct(e, "type", w));
  }
  for (const y in r) {
    let x = r[y];
    if (c && y === "value" && x == null) {
      e.value = e.__value = "", l[y] = x;
      continue;
    }
    if (y === "class") {
      var d = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      tt(e, d, x, n, t?.[At], r[At]), l[y] = x, l[At] = r[At];
      continue;
    }
    if (y === "style") {
      Oe(e, x, t?.[lt], r[lt]), l[y] = x, l[lt] = r[lt];
      continue;
    }
    var m = l[y];
    if (!(x === m && !(x === void 0 && e.hasAttribute(y)))) {
      l[y] = x;
      var P = y[0] + y[1];
      if (P !== "$$")
        if (P === "on") {
          const H = {}, O = "$$" + y;
          let B = y.slice(2);
          var h = ta(B);
          if (Qi(B) && (B = B.slice(0, -7), H.capture = !0), !h && m) {
            if (x != null) continue;
            e.removeEventListener(B, l[O], H), l[O] = null;
          }
          if (h)
            jn(B, e, x), Zt([B]);
          else if (x != null) {
            let G = function(U) {
              l[y].call(this, U);
            };
            l[O] = Ki(B, e, G, H);
          }
        } else if (y === "style")
          ct(e, y, x);
        else if (y === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!x
          );
        else if (!s && (y === "__value" || y === "value" && x != null))
          e.value = e.__value = x;
        else if (y === "selected" && c)
          Wa(
            /** @type {HTMLOptionElement} */
            e,
            x
          );
        else {
          var b = y;
          u || (b = $i(b));
          var T = b === "defaultValue" || b === "defaultChecked";
          if (x == null && !s && !T)
            if (o[y] = null, b === "value" || b === "checked") {
              let H = (
                /** @type {HTMLInputElement} */
                e
              );
              const O = t === void 0;
              if (b === "value") {
                let B = H.defaultValue;
                H.removeAttribute(b), H.defaultValue = B, H.value = H.__value = O ? B : null;
              } else {
                let B = H.defaultChecked;
                H.removeAttribute(b), H.defaultChecked = B, H.checked = O ? B : !1;
              }
            } else
              e.removeAttribute(y);
          else T || g.includes(b) && (s || typeof x != "string") ? (e[b] = x, b in o && (o[b] = ea)) : typeof x != "function" && ct(e, b, x);
        }
    }
  }
  return l;
}
function Ya(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Yi(i, r, n, (u) => {
    var l = void 0, c = {}, v = e.nodeName === qa, g = !1;
    if (Gn(() => {
      var d = t(...u.map(f)), m = Za(
        e,
        l,
        d,
        a,
        o,
        s
      );
      g && v && "value" in d && Hr(
        /** @type {HTMLSelectElement} */
        e,
        d.value
      );
      for (let h of Object.getOwnPropertySymbols(c))
        d[h] || et(c[h]);
      for (let h of Object.getOwnPropertySymbols(d)) {
        var P = d[h];
        h.description === Ji && (!l || P !== l[h]) && (c[h] && et(c[h]), c[h] = nt(() => Da(e, () => P))), m[h] = P;
      }
      l = m;
    }), v) {
      var w = (
        /** @type {HTMLSelectElement} */
        e
      );
      jr(() => {
        Hr(
          w,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), Va(w);
      });
    }
    g = !0;
  });
}
function Qn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[qi] ??= {
      [Yn]: e.nodeName.includes("-"),
      [Jn]: e.namespaceURI === Wi
    }
  );
}
var hn = /* @__PURE__ */ new Map();
function Kn(e) {
  var t = e.getAttribute("is") || e.nodeName, r = hn.get(t);
  if (r) return r;
  hn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = Zi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Nn(i);
  }
  return r;
}
function cr(e, t) {
  return e === t || e?.[zr] === t;
}
function qr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    ft.r
  ), a = (
    /** @type {Effect} */
    qt
  );
  return jr(() => {
    var o, s;
    return ra(() => {
      o = s, s = [], ie(() => {
        cr(r(...s), e) || (t(e, ...s), o && cr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & na; )
        u = u.parent;
      const l = () => {
        s && cr(r(...s), e) && t(null, ...s);
      }, c = u.teardown;
      u.teardown = () => {
        l(), c?.();
      };
    };
  }), e;
}
function Ja(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    ft
  ), r = t.l.u;
  if (!r) return;
  let n = () => Ee(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Er(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => f(o);
  }
  r.b.length && ia(() => {
    dn(t, n), xr(r.b);
  }), we(() => {
    const i = ie(() => r.m.map(aa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && we(() => {
    dn(t, n), xr(r.a);
  });
}
function dn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) f(r);
  t();
}
const Qa = {
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
function Ka(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Qa
  );
}
const $a = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Et(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Et(i) && (i = i());
      const a = wr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Et(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = wr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === zr || t === Vn) return !1;
    for (let r of e.props)
      if (Et(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Et(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function es(...e) {
  return new Proxy({ props: e }, $a);
}
function M(e, t, r, n) {
  var i = !Dn || (r & ua) !== 0, a = (r & la) !== 0, o = (r & ca) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), c = () => o && i ? (l ??= Er(
    /** @type {() => V} */
    n
  ), f(l)) : (u && (u = !1, s = o ? ie(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let v;
  if (a) {
    var g = zr in e || Vn in e;
    v = wr(e, t)?.set ?? (g && t in e ? (y) => e[t] = y : void 0);
  }
  var w, d = !1;
  a ? [w, d] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : w = /** @type {V} */
  e[t], w === void 0 && n !== void 0 && (w = c(), v && (i && sa(), v(w)));
  var m;
  if (i ? m = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y === void 0 ? c() : (u = !0, y);
  } : m = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y !== void 0 && (s = /** @type {V} */
    void 0), y === void 0 ? s : y;
  }, i && (r & oa) === 0)
    return m;
  if (v) {
    var P = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(y, x) {
        return arguments.length > 0 ? ((!i || !x || P || d) && v(x ? m() : y), y) : m();
      })
    );
  }
  var h = !1, b = ((r & fa) !== 0 ? Er : Un)(() => (h = !1, m()));
  a && f(b);
  var T = (
    /** @type {Effect} */
    qt
  );
  return (
    /** @type {() => V} */
    (function(y, x) {
      if (arguments.length > 0) {
        const H = x ? f(b) : i && a ? Ht(y) : y;
        return A(b, H), h = !0, s !== void 0 && (s = H), y;
      }
      return ha && h || (T.f & Fn) !== 0 ? b.v : f(b);
    })
  );
}
const ts = [
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
], mn = {
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
ts.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: mn[t][r],
    secondary: mn[t][n]
  }
}), {});
function rs(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var hr, pn;
function ns() {
  if (pn) return hr;
  pn = 1;
  var e = function(b) {
    return t(b) && !r(b);
  };
  function t(h) {
    return !!h && typeof h == "object";
  }
  function r(h) {
    var b = Object.prototype.toString.call(h);
    return b === "[object RegExp]" || b === "[object Date]" || a(h);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(h) {
    return h.$$typeof === i;
  }
  function o(h) {
    return Array.isArray(h) ? [] : {};
  }
  function s(h, b) {
    return b.clone !== !1 && b.isMergeableObject(h) ? m(o(h), h, b) : h;
  }
  function u(h, b, T) {
    return h.concat(b).map(function(y) {
      return s(y, T);
    });
  }
  function l(h, b) {
    if (!b.customMerge)
      return m;
    var T = b.customMerge(h);
    return typeof T == "function" ? T : m;
  }
  function c(h) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(h).filter(function(b) {
      return Object.propertyIsEnumerable.call(h, b);
    }) : [];
  }
  function v(h) {
    return Object.keys(h).concat(c(h));
  }
  function g(h, b) {
    try {
      return b in h;
    } catch {
      return !1;
    }
  }
  function w(h, b) {
    return g(h, b) && !(Object.hasOwnProperty.call(h, b) && Object.propertyIsEnumerable.call(h, b));
  }
  function d(h, b, T) {
    var y = {};
    return T.isMergeableObject(h) && v(h).forEach(function(x) {
      y[x] = s(h[x], T);
    }), v(b).forEach(function(x) {
      w(h, x) || (g(h, x) && T.isMergeableObject(b[x]) ? y[x] = l(x, T)(h[x], b[x], T) : y[x] = s(b[x], T));
    }), y;
  }
  function m(h, b, T) {
    T = T || {}, T.arrayMerge = T.arrayMerge || u, T.isMergeableObject = T.isMergeableObject || e, T.cloneUnlessOtherwiseSpecified = s;
    var y = Array.isArray(b), x = Array.isArray(h), H = y === x;
    return H ? y ? T.arrayMerge(h, b, T) : d(h, b, T) : s(b, T);
  }
  m.all = function(b, T) {
    if (!Array.isArray(b))
      throw new Error("first argument should be an array");
    return b.reduce(function(y, x) {
      return m(y, x, T);
    }, {});
  };
  var P = m;
  return hr = P, hr;
}
var is = ns();
const as = /* @__PURE__ */ rs(is);
var Mr = function(e, t) {
  return Mr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Mr(e, t);
};
function Kt(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Mr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var F = function() {
  return F = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, F.apply(this, arguments);
};
function ss(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function dr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function mr(e, t) {
  var r = t && t.cache ? t.cache : ds, n = t && t.serializer ? t.serializer : cs, i = t && t.strategy ? t.strategy : us;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function os(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function ls(e, t, r, n) {
  var i = os(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function $n(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function ei(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function us(e, t) {
  var r = e.length === 1 ? ls : $n;
  return ei(e, this, r, t.cache.create(), t.serializer);
}
function fs(e, t) {
  return ei(e, this, $n, t.cache.create(), t.serializer);
}
var cs = function() {
  return JSON.stringify(arguments);
}, hs = (
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
), ds = {
  create: function() {
    return new hs();
  }
}, pr = {
  variadic: fs
}, C;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(C || (C = {}));
var q;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(q || (q = {}));
var dt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(dt || (dt = {}));
function vn(e) {
  return e.type === q.literal;
}
function ms(e) {
  return e.type === q.argument;
}
function ti(e) {
  return e.type === q.number;
}
function ri(e) {
  return e.type === q.date;
}
function ni(e) {
  return e.type === q.time;
}
function ii(e) {
  return e.type === q.select;
}
function ai(e) {
  return e.type === q.plural;
}
function ps(e) {
  return e.type === q.pound;
}
function si(e) {
  return e.type === q.tag;
}
function oi(e) {
  return !!(e && typeof e == "object" && e.type === dt.number);
}
function Pr(e) {
  return !!(e && typeof e == "object" && e.type === dt.dateTime);
}
var li = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, vs = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function gs(e) {
  var t = {};
  return e.replace(vs, function(r) {
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
var bs = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function _s(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(bs).filter(function(g) {
    return g.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, c = u; l < c.length; l++) {
      var v = c[l];
      if (v.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function ys(e) {
  return e.replace(/^(.*?)-/, "");
}
var gn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, ui = /^(@+)?(\+|#+)?[rs]?$/g, xs = /(\*)(0+)|(#+)(0+)|(0+)/g, fi = /^(0+)$/;
function bn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(ui, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function ci(e) {
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
function Es(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !fi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function _n(e) {
  var t = {}, r = ci(e);
  return r || t;
}
function ws(e) {
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
        t.style = "unit", t.unit = ys(i.options[0]);
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
        t = F(F(F({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return F(F({}, u), _n(l));
        }, {}));
        continue;
      case "engineering":
        t = F(F(F({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return F(F({}, u), _n(l));
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
        i.options[0].replace(xs, function(u, l, c, v, g, w) {
          if (l)
            t.minimumIntegerDigits = c.length;
          else {
            if (v && g)
              throw new Error("We currently do not support maximum integer digits");
            if (w)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (fi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (gn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(gn, function(u, l, c, v, g, w) {
        return c === "*" ? t.minimumFractionDigits = l.length : v && v[0] === "#" ? t.maximumFractionDigits = v.length : g && w ? (t.minimumFractionDigits = g.length, t.maximumFractionDigits = g.length + w.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = F(F({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = F(F({}, t), bn(a)));
      continue;
    }
    if (ui.test(i.stem)) {
      t = F(F({}, t), bn(i.stem));
      continue;
    }
    var o = ci(i.stem);
    o && (t = F(F({}, t), o));
    var s = Es(i.stem);
    s && (t = F(F({}, t), s));
  }
  return t;
}
var kt = {
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
function Ts(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), u = "a", l = Ss(t);
      for ((l == "H" || l == "k") && (s = 0); s-- > 0; )
        r += u;
      for (; o-- > 0; )
        r = l + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Ss(e) {
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
  var i = kt[n || ""] || kt[r || ""] || kt["".concat(r, "-001")] || kt["001"];
  return i[0];
}
var vr, As = new RegExp("^".concat(li.source, "*")), Hs = new RegExp("".concat(li.source, "*$"));
function k(e, t) {
  return { start: e, end: t };
}
var Ms = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ps = !!String.fromCodePoint, Os = !!Object.fromEntries, Bs = !!String.prototype.codePointAt, Is = !!String.prototype.trimStart, Ns = !!String.prototype.trimEnd, Ls = !!Number.isSafeInteger, Cs = Ls ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Or = !0;
try {
  var Rs = di("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Or = ((vr = Rs.exec("a")) === null || vr === void 0 ? void 0 : vr[0]) === "a";
} catch {
  Or = !1;
}
var yn = Ms ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Br = Ps ? String.fromCodePoint : (
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
), xn = (
  // native
  Os ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), hi = Bs ? (
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
), ks = Is ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(As, "");
  }
), Ds = Ns ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Hs, "");
  }
);
function di(e, t) {
  return new RegExp(e, t);
}
var Ir;
if (Or) {
  var En = di("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Ir = function(t, r) {
    var n;
    En.lastIndex = r;
    var i = En.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Ir = function(t, r) {
    for (var n = []; ; ) {
      var i = hi(t, r);
      if (i === void 0 || mi(i) || js(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Br.apply(void 0, n);
  };
var Us = (
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
            var s = this.clonePosition();
            this.bump(), i.push({
              type: q.pound,
              location: k(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(C.UNMATCHED_CLOSING_TAG, k(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Nr(this.peek() || 0)) {
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
            type: q.literal,
            value: "<".concat(i, "/>"),
            location: k(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Nr(this.char()))
            return this.error(C.INVALID_TAG, k(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(C.UNMATCHED_CLOSING_TAG, k(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: q.tag,
              value: i,
              children: o,
              location: k(n, this.clonePosition())
            },
            err: null
          } : this.error(C.INVALID_TAG, k(s, this.clonePosition())));
        } else
          return this.error(C.UNCLOSED_TAG, k(n, this.clonePosition()));
      } else
        return this.error(C.INVALID_TAG, k(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Gs(this.char()); )
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
        var s = this.tryParseLeftAngleBracket();
        if (s) {
          i += s;
          continue;
        }
        break;
      }
      var u = k(n, this.clonePosition());
      return {
        val: { type: q.literal, value: i, location: u },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Fs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Br.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Br(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(C.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(C.EMPTY_ARGUMENT, k(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(C.MALFORMED_ARGUMENT, k(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(C.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: q.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: k(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(C.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(C.MALFORMED_ARGUMENT, k(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Ir(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = k(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(C.EXPECT_ARGUMENT_TYPE, k(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var c = this.clonePosition(), v = this.parseSimpleArgStyleIfPossible();
            if (v.err)
              return v;
            var g = Ds(v.val);
            if (g.length === 0)
              return this.error(C.EXPECT_ARGUMENT_STYLE, k(this.clonePosition(), this.clonePosition()));
            var w = k(c, this.clonePosition());
            l = { style: g, styleLocation: w };
          }
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var m = k(i, this.clonePosition());
          if (l && yn(l?.style, "::", 0)) {
            var P = ks(l.style.slice(2));
            if (s === "number") {
              var v = this.parseNumberSkeletonFromString(P, l.styleLocation);
              return v.err ? v : {
                val: { type: q.number, value: n, location: m, style: v.val },
                err: null
              };
            } else {
              if (P.length === 0)
                return this.error(C.EXPECT_DATE_TIME_SKELETON, m);
              var h = P;
              this.locale && (h = Ts(P, this.locale));
              var g = {
                type: dt.dateTime,
                pattern: h,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? gs(h) : {}
              }, b = s === "date" ? q.date : q.time;
              return {
                val: { type: b, value: n, location: m, style: g },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? q.number : s === "date" ? q.date : q.time,
              value: n,
              location: m,
              style: (a = l?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var T = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(C.EXPECT_SELECT_ARGUMENT_OPTIONS, k(T, F({}, T)));
          this.bumpSpace();
          var y = this.parseIdentifierIfPossible(), x = 0;
          if (s !== "select" && y.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(C.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, k(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var v = this.tryParseDecimalInteger(C.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, C.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (v.err)
              return v;
            this.bumpSpace(), y = this.parseIdentifierIfPossible(), x = v.val;
          }
          var H = this.tryParsePluralOrSelectOptions(t, s, r, y);
          if (H.err)
            return H;
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var O = k(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: q.select,
              value: n,
              options: xn(H.val),
              location: O
            },
            err: null
          } : {
            val: {
              type: q.plural,
              value: n,
              options: xn(H.val),
              offset: x,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: O
            },
            err: null
          };
        }
        default:
          return this.error(C.INVALID_ARGUMENT_TYPE, k(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(C.EXPECT_ARGUMENT_CLOSING_BRACE, k(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(C.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, k(i, this.clonePosition()));
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
        n = _s(t);
      } catch {
        return this.error(C.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: dt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? ws(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, c = i.location; ; ) {
        if (l.length === 0) {
          var v = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var g = this.tryParseDecimalInteger(C.EXPECT_PLURAL_ARGUMENT_SELECTOR, C.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (g.err)
              return g;
            c = k(v, this.clonePosition()), l = this.message.slice(v.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? C.DUPLICATE_SELECT_ARGUMENT_SELECTOR : C.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, c);
        l === "other" && (o = !0), this.bumpSpace();
        var w = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? C.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : C.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, k(this.clonePosition(), this.clonePosition()));
        var d = this.parseMessage(t + 1, r, n);
        if (d.err)
          return d;
        var m = this.tryParseArgumentClose(w);
        if (m.err)
          return m;
        s.push([
          l,
          {
            value: d.val,
            location: k(w, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, c = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? C.EXPECT_SELECT_ARGUMENT_SELECTOR : C.EXPECT_PLURAL_ARGUMENT_SELECTOR, k(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(C.MISSING_OTHER_CLAUSE, k(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var a = !1, o = 0; !this.isEOF(); ) {
        var s = this.char();
        if (s >= 48 && s <= 57)
          a = !0, o = o * 10 + (s - 48), this.bump();
        else
          break;
      }
      var u = k(i, this.clonePosition());
      return a ? (o *= n, Cs(o) ? { val: o, err: null } : this.error(r, u)) : this.error(t, u);
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
      var r = hi(this.message, t);
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
      if (yn(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && mi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Nr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Fs(e) {
  return Nr(e) || e === 47;
}
function Gs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function mi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function js(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Lr(e) {
  e.forEach(function(t) {
    if (delete t.location, ii(t) || ai(t))
      for (var r in t.options)
        delete t.options[r].location, Lr(t.options[r].value);
    else ti(t) && oi(t.style) || (ri(t) || ni(t)) && Pr(t.style) ? delete t.style.location : si(t) && Lr(t.children);
  });
}
function Vs(e, t) {
  t === void 0 && (t = {}), t = F({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Us(e, t).parse();
  if (r.err) {
    var n = SyntaxError(C[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Lr(r.val), r.val;
}
var mt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(mt || (mt = {}));
var $t = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), wn = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), mt.INVALID_VALUE, a) || this;
    }
    return t;
  })($t)
), zs = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), mt.INVALID_VALUE, i) || this;
    }
    return t;
  })($t)
), Xs = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), mt.MISSING_VALUE, n) || this;
    }
    return t;
  })($t)
), me;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(me || (me = {}));
function qs(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== me.literal || r.type !== me.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Ws(e) {
  return typeof e == "function";
}
function Ft(e, t, r, n, i, a, o) {
  if (e.length === 1 && vn(e[0]))
    return [
      {
        type: me.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var c = l[u];
    if (vn(c)) {
      s.push({
        type: me.literal,
        value: c.value
      });
      continue;
    }
    if (ps(c)) {
      typeof a == "number" && s.push({
        type: me.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var v = c.value;
    if (!(i && v in i))
      throw new Xs(v, o);
    var g = i[v];
    if (ms(c)) {
      (!g || typeof g == "string" || typeof g == "number") && (g = typeof g == "string" || typeof g == "number" ? String(g) : ""), s.push({
        type: typeof g == "string" ? me.literal : me.object,
        value: g
      });
      continue;
    }
    if (ri(c)) {
      var w = typeof c.style == "string" ? n.date[c.style] : Pr(c.style) ? c.style.parsedOptions : void 0;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, w).format(g)
      });
      continue;
    }
    if (ni(c)) {
      var w = typeof c.style == "string" ? n.time[c.style] : Pr(c.style) ? c.style.parsedOptions : n.time.medium;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, w).format(g)
      });
      continue;
    }
    if (ti(c)) {
      var w = typeof c.style == "string" ? n.number[c.style] : oi(c.style) ? c.style.parsedOptions : void 0;
      w && w.scale && (g = g * (w.scale || 1)), s.push({
        type: me.literal,
        value: r.getNumberFormat(t, w).format(g)
      });
      continue;
    }
    if (si(c)) {
      var d = c.children, m = c.value, P = i[m];
      if (!Ws(P))
        throw new zs(m, "function", o);
      var h = Ft(d, t, r, n, i, a), b = P(h.map(function(x) {
        return x.value;
      }));
      Array.isArray(b) || (b = [b]), s.push.apply(s, b.map(function(x) {
        return {
          type: typeof x == "string" ? me.literal : me.object,
          value: x
        };
      }));
    }
    if (ii(c)) {
      var T = c.options[g] || c.options.other;
      if (!T)
        throw new wn(c.value, g, Object.keys(c.options), o);
      s.push.apply(s, Ft(T.value, t, r, n, i));
      continue;
    }
    if (ai(c)) {
      var T = c.options["=".concat(g)];
      if (!T) {
        if (!Intl.PluralRules)
          throw new $t(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, mt.MISSING_INTL_API, o);
        var y = r.getPluralRules(t, { type: c.pluralType }).select(g - (c.offset || 0));
        T = c.options[y] || c.options.other;
      }
      if (!T)
        throw new wn(c.value, g, Object.keys(c.options), o);
      s.push.apply(s, Ft(T.value, t, r, n, i, g - (c.offset || 0)));
      continue;
    }
  }
  return qs(s);
}
function Zs(e, t) {
  return t ? F(F(F({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = F(F({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ys(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Zs(e[n], t[n]), r;
  }, F({}, e)) : e;
}
function gr(e) {
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
function Js(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, dr([void 0], r, !1)))();
    }, {
      cache: gr(e.number),
      strategy: pr.variadic
    }),
    getDateTimeFormat: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, dr([void 0], r, !1)))();
    }, {
      cache: gr(e.dateTime),
      strategy: pr.variadic
    }),
    getPluralRules: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, dr([void 0], r, !1)))();
    }, {
      cache: gr(e.pluralRules),
      strategy: pr.variadic
    })
  };
}
var Qs = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var a = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(u) {
        var l = a.formatToParts(u);
        if (l.length === 1)
          return l[0].value;
        var c = l.reduce(function(v, g) {
          return !v.length || g.type !== me.literal || typeof v[v.length - 1] != "string" ? v.push(g.value) : v[v.length - 1] += g.value, v;
        }, []);
        return c.length <= 1 ? c[0] || "" : c;
      }, this.formatToParts = function(u) {
        return Ft(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
      }, this.resolvedOptions = function() {
        var u;
        return {
          locale: ((u = a.resolvedLocale) === null || u === void 0 ? void 0 : u.toString()) || Intl.NumberFormat.supportedLocalesOf(a.locales)[0]
        };
      }, this.getAst = function() {
        return a.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var o = i || {};
        o.formatters;
        var s = ss(o, ["formatters"]);
        this.ast = e.__parse(t, F(F({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Ys(e.formats, n), this.formatters = i && i.formatters || Js(this.formatterCache);
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
    }, e.__parse = Vs, e.formats = {
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
function Ks(e, t) {
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
const We = {}, $s = (e, t, r) => r && (t in We || (We[t] = {}), e in We[t] || (We[t][e] = r), r), pi = (e, t) => {
  if (t == null)
    return;
  if (t in We && e in We[t])
    return We[t][e];
  const r = er(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = to(i, e);
    if (a)
      return $s(e, t, a);
  }
};
let Wr;
const Bt = Ot({});
function eo(e) {
  return Wr[e] || null;
}
function vi(e) {
  return e in Wr;
}
function to(e, t) {
  if (!vi(e))
    return null;
  const r = eo(e);
  return Ks(r, t);
}
function ro(e) {
  if (e == null)
    return;
  const t = er(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (vi(n))
      return n;
  }
}
function no(e, ...t) {
  delete We[e], Bt.update((r) => (r[e] = as.all([r[e] || {}, ...t]), r));
}
vt(
  [Bt],
  ([e]) => Object.keys(e)
);
Bt.subscribe((e) => Wr = e);
const Gt = {};
function io(e, t) {
  Gt[e].delete(t), Gt[e].size === 0 && delete Gt[e];
}
function gi(e) {
  return Gt[e];
}
function ao(e) {
  return er(e).map((t) => {
    const r = gi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Cr(e) {
  return e == null ? !1 : er(e).some(
    (t) => {
      var r;
      return (r = gi(t)) == null ? void 0 : r.size;
    }
  );
}
function so(e, t) {
  return Promise.all(
    t.map((n) => (io(e, n), n().then((i) => i.default || i)))
  ).then((n) => no(e, ...n));
}
const Tt = {};
function bi(e) {
  if (!Cr(e))
    return e in Tt ? Tt[e] : Promise.resolve();
  const t = ao(e);
  return Tt[e] = Promise.all(
    t.map(
      ([r, n]) => so(r, n)
    )
  ).then(() => {
    if (Cr(e))
      return bi(e);
    delete Tt[e];
  }), Tt[e];
}
const oo = {
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
}, lo = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: oo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, uo = lo;
function pt() {
  return uo;
}
const br = Ot(!1);
var fo = Object.defineProperty, co = Object.defineProperties, ho = Object.getOwnPropertyDescriptors, Tn = Object.getOwnPropertySymbols, mo = Object.prototype.hasOwnProperty, po = Object.prototype.propertyIsEnumerable, Sn = (e, t, r) => t in e ? fo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, vo = (e, t) => {
  for (var r in t || (t = {}))
    mo.call(t, r) && Sn(e, r, t[r]);
  if (Tn)
    for (var r of Tn(t))
      po.call(t, r) && Sn(e, r, t[r]);
  return e;
}, go = (e, t) => co(e, ho(t));
let Rr;
const zt = Ot(null);
function An(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function er(e, t = pt().fallbackLocale) {
  const r = An(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...An(t)])] : r;
}
function it() {
  return Rr ?? void 0;
}
zt.subscribe((e) => {
  Rr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const bo = (e) => {
  if (e && ro(e) && Cr(e)) {
    const { loadingDelay: t } = pt();
    let r;
    return typeof window < "u" && it() != null && t ? r = window.setTimeout(
      () => br.set(!0),
      t
    ) : br.set(!0), bi(e).then(() => {
      zt.set(e);
    }).finally(() => {
      clearTimeout(r), br.set(!1);
    });
  }
  return zt.set(e);
}, gt = go(vo({}, zt), {
  set: bo
}), tr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var _o = Object.defineProperty, Xt = Object.getOwnPropertySymbols, _i = Object.prototype.hasOwnProperty, yi = Object.prototype.propertyIsEnumerable, Hn = (e, t, r) => t in e ? _o(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Zr = (e, t) => {
  for (var r in t || (t = {}))
    _i.call(t, r) && Hn(e, r, t[r]);
  if (Xt)
    for (var r of Xt(t))
      yi.call(t, r) && Hn(e, r, t[r]);
  return e;
}, bt = (e, t) => {
  var r = {};
  for (var n in e)
    _i.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Xt)
    for (var n of Xt(e))
      t.indexOf(n) < 0 && yi.call(e, n) && (r[n] = e[n]);
  return r;
};
const Mt = (e, t) => {
  const { formats: r } = pt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, yo = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = bt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Mt("number", n)), new Intl.NumberFormat(r, i);
  }
), xo = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = bt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Mt("date", n) : Object.keys(i).length === 0 && (i = Mt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Eo = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = bt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Mt("time", n) : Object.keys(i).length === 0 && (i = Mt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), wo = (e = {}) => {
  var t = e, {
    locale: r = it()
  } = t, n = bt(t, [
    "locale"
  ]);
  return yo(Zr({ locale: r }, n));
}, To = (e = {}) => {
  var t = e, {
    locale: r = it()
  } = t, n = bt(t, [
    "locale"
  ]);
  return xo(Zr({ locale: r }, n));
}, So = (e = {}) => {
  var t = e, {
    locale: r = it()
  } = t, n = bt(t, [
    "locale"
  ]);
  return Eo(Zr({ locale: r }, n));
}, Ao = tr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = it()) => new Qs(e, t, pt().formats, {
    ignoreTag: pt().ignoreTag
  })
), Ho = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = it(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let c = pi(e, u);
  if (!c)
    c = (a = (i = (n = (r = pt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof c != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof c}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), c;
  if (!s)
    return c;
  let v = c;
  try {
    v = Ao(c, u).format(s);
  } catch (g) {
    g instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      g.message
    );
  }
  return v;
}, Mo = (e, t) => So(t).format(e), Po = (e, t) => To(t).format(e), Oo = (e, t) => wo(t).format(e), Bo = (e, t = it()) => pi(e, t);
vt([gt, Bt], () => Ho);
vt([gt], () => Mo);
vt([gt], () => Po);
vt([gt], () => Oo);
vt([gt, Bt], () => Bo);
const Io = "__i18n__", No = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Lo = [
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
function Co(e) {
  return typeof e == "string" && e.includes(Io);
}
class Ro {
  load_component;
  #t = W(Ht({}));
  get shared() {
    return f(this.#t);
  }
  set shared(t) {
    A(this.#t, t, !0);
  }
  #r = W(Ht({}));
  get props() {
    return f(this.#r);
  }
  set props(t) {
    A(this.#r, t, !0);
  }
  #e = W((t) => t);
  get i18n() {
    return f(this.#e);
  }
  set i18n(t) {
    A(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Lo;
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
    for (const n of No)
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
    ), we(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), ie(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && gt.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [a, o] = n.split("."), s = this.i18n(i);
        a === "shared" ? this.shared[o] = s : this.props[o] = s;
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
    return ba(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = Co(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    we(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
da();
var ko = /* @__PURE__ */ qn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Mn = /* @__PURE__ */ ue("<!> <!>", 1), Do = /* @__PURE__ */ ue('<div class="placeholder svelte-1stq1b1"></div>');
function Uo(e, t) {
  Jt(t, !1);
  let r = M(t, "height", 8, void 0), n = M(t, "min_height", 8, void 0), i = M(t, "max_height", 8, void 0), a = M(t, "width", 8, void 0), o = M(t, "elem_id", 8, ""), s = M(t, "elem_classes", 24, () => []), u = M(t, "variant", 8, "solid"), l = M(t, "border_mode", 8, "base"), c = M(t, "padding", 8, !0), v = M(t, "type", 8, "normal"), g = M(t, "test_id", 8, void 0), w = M(t, "explicit_call", 8, !1), d = M(t, "container", 8, !0), m = M(t, "visible", 8, !0), P = M(t, "allow_overflow", 8, !0), h = M(t, "overflow_behavior", 8, "auto"), b = M(t, "scale", 8, null), T = M(t, "min_width", 8, 0), y = M(t, "flex", 12, !1), x = M(t, "resizable", 8, !1), H = M(t, "rtl", 8, !1), O = M(t, "fullscreen", 12, !1), B = M(t, "label", 8, void 0), G = $e(O()), U = $e(), fe = v() === "fieldset" ? "fieldset" : "div", K = $e(0), Z = $e(0), z = $e(null);
  function Be(ae) {
    O() && ae.key === "Escape" && O(!1);
  }
  const Ie = (ae) => {
    if (ae !== void 0) {
      if (typeof ae == "number")
        return ae + "px";
      if (typeof ae == "string")
        return ae;
    }
  }, je = (ae) => {
    let Te = ae.clientY;
    const le = (ne) => {
      const _e = ne.clientY - Te;
      Te = ne.clientY, pa(U, f(U).style.height = `${f(U).offsetHeight + _e}px`);
    }, Ne = () => {
      window.removeEventListener("mousemove", le), window.removeEventListener("mouseup", Ne);
    };
    window.addEventListener("mousemove", le), window.addEventListener("mouseup", Ne);
  };
  nn(
    () => (Ee(O()), f(G), f(U)),
    () => {
      O() !== f(G) && (A(G, O()), O() ? (A(z, f(U).getBoundingClientRect()), A(K, f(U).offsetHeight), A(Z, f(U).offsetWidth), window.addEventListener("keydown", Be)) : (A(z, null), window.removeEventListener("keydown", Be)));
    }
  ), nn(() => Ee(m()), () => {
    m() || y(!1);
  }), ma(), Ja();
  var Ye = ut(), Ve = de(Ye);
  {
    var ke = (ae) => {
      var Te = Mn(), le = de(Te);
      ka(le, () => fe, !1, (_e, De) => {
        qr(_e, (ye) => A(U, ye), () => f(U)), Ya(
          _e,
          (ye, pe) => ({
            "data-testid": g(),
            id: o(),
            class: `block ${ye ?? ""}`,
            dir: H() ? "rtl" : "ltr",
            "aria-label": B(),
            style: "",
            [At]: {
              hidden: m() === "hidden",
              padded: c(),
              flex: y(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !w() && !d(),
              fullscreen: O(),
              animating: O() && f(z) !== null,
              "auto-margin": b() === null
            },
            [lt]: pe
          }),
          [
            () => (Ee(s()), ie(() => s()?.join(" ") || "")),
            () => ({
              height: (Ee(O()), Ee(r()), ie(() => O() ? void 0 : Ie(r()))),
              "min-height": (Ee(O()), Ee(n()), ie(() => O() ? void 0 : Ie(n()))),
              "max-height": (Ee(O()), Ee(i()), ie(() => O() ? void 0 : Ie(i()))),
              "--start-top": (f(z), ie(() => f(z) ? `${f(z).top}px` : "0px")),
              "--start-left": (f(z), ie(() => f(z) ? `${f(z).left}px` : "0px")),
              "--start-width": (f(z), ie(() => f(z) ? `${f(z).width}px` : "0px")),
              "--start-height": (f(z), ie(() => f(z) ? `${f(z).height}px` : "0px")),
              width: (Ee(O()), Ee(a()), ie(() => O() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : Ie(a()))),
              "border-style": u(),
              overflow: P() ? h() : "hidden",
              "flex-grow": b(),
              "min-width": `calc(min(${T()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Se = Mn(), ze = de(Se);
        Ar(ze, t, "default", {});
        var Le = V(ze, 2);
        {
          var Je = (ye) => {
            var pe = ko();
            He("mousedown", pe, je), R(ye, pe);
          };
          Q(Le, (ye) => {
            x() && ye(Je);
          });
        }
        R(De, Se);
      });
      var Ne = V(le, 2);
      {
        var ne = (_e) => {
          var De = Do();
          let Se;
          $(() => Se = Oe(De, "", Se, {
            height: f(K) + "px",
            width: f(Z) + "px"
          })), R(_e, De);
        };
        Q(Ne, (_e) => {
          O() && _e(ne);
        });
      }
      R(ae, Te);
    };
    Q(Ve, (ae) => {
      (m() === !0 || m() === "hidden") && ae(ke);
    });
  }
  R(e, Ye), Yt();
}
var Fo = /* @__PURE__ */ ue('<span class="svelte-vvirtv"> </span>'), Go = /* @__PURE__ */ ue("<button><!> <div><!> <!></div></button>");
function Pn(e, t) {
  let r = M(t, "label", 3, ""), n = M(t, "show_label", 3, !1), i = M(t, "pending", 3, !1), a = M(t, "size", 3, "small"), o = M(t, "padded", 3, !0), s = M(t, "highlight", 3, !1), u = M(t, "disabled", 3, !1), l = M(t, "hasPopup", 3, !1), c = M(t, "color", 3, "var(--block-label-text-color)"), v = M(t, "transparent", 3, !1), g = M(t, "background", 3, "var(--block-background-fill)"), w = M(t, "border", 3, "transparent"), d = Me(() => s() ? "var(--color-accent)" : c());
  var m = Go();
  let P, h;
  var b = re(m);
  {
    var T = (G) => {
      var U = Fo(), fe = re(U);
      $(() => he(fe, r())), R(G, U);
    };
    Q(b, (G) => {
      n() && G(T);
    });
  }
  var y = V(b, 2);
  let x;
  var H = re(y);
  La(H, () => t.Icon, (G, U) => {
    U(G, {});
  });
  var O = V(H, 2);
  {
    var B = (G) => {
      var U = ut(), fe = de(U);
      Ha(fe, () => t.children), R(G, U);
    };
    Q(O, (G) => {
      t.children && G(B);
    });
  }
  $(() => {
    P = tt(m, 1, "icon-button svelte-vvirtv", null, P, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: v()
    }), m.disabled = u(), ct(m, "aria-label", r()), ct(m, "aria-haspopup", l()), ct(m, "title", r()), h = Oe(m, "", h, {
      "--border-color": w(),
      color: !u() && f(d) ? f(d) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : g()
    }), x = tt(y, 1, "svelte-vvirtv", null, x, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), jn("click", m, function(...G) {
    t.onclick?.apply(this, G);
  }), R(e, m);
}
Zt(["click"]);
var jo = /* @__PURE__ */ qn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function On(e) {
  var t = jo();
  R(e, t);
}
Zt(["click"]);
function _r(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Bn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function kr(e, t, r, n) {
  if (typeof r == "number" || Bn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Bn(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          kr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = kr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function In(e, t = {}) {
  const r = Ot(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), c = (
    /** @type {T | undefined} */
    e
  ), v = 1, g = 0, w = !1;
  function d(P, h = {}) {
    c = P;
    const b = u = {};
    return e == null || h.hard || m.stiffness >= 1 && m.damping >= 1 ? (w = !0, o = Pe.now(), l = P, r.set(e = c), Promise.resolve()) : (h.soft && (g = 1 / ((h.soft === !0 ? 0.5 : +h.soft) * 60), v = 0), s || (o = Pe.now(), w = !1, s = Ra((T) => {
      if (w)
        return w = !1, s = null, !1;
      v = Math.min(v + g, 1);
      const y = Math.min(T - o, 1e3 / 30), x = {
        inv_mass: v,
        opts: m,
        settled: !0,
        dt: y * 60 / 1e3
      }, H = kr(x, l, e, c);
      return o = T, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      H), x.settled && (s = null), !x.settled;
    })), new Promise((T) => {
      s.promise.then(() => {
        b === u && T();
      });
    }));
  }
  const m = {
    set: d,
    update: (P, h) => d(P(
      /** @type {T} */
      c,
      /** @type {T} */
      e
    ), h),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return m;
}
var Vo = /* @__PURE__ */ ue('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function zo(e, t) {
  Jt(t, !0);
  const r = () => an(u, "$top", i), n = () => an(l, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(T, y, x, H) {
    function O(B) {
      return B instanceof x ? B : new x(function(G) {
        G(B);
      });
    }
    return new (x || (x = Promise))(function(B, G) {
      function U(Z) {
        try {
          K(H.next(Z));
        } catch (z) {
          G(z);
        }
      }
      function fe(Z) {
        try {
          K(H.throw(Z));
        } catch (z) {
          G(z);
        }
      }
      function K(Z) {
        Z.done ? B(Z.value) : O(Z.value).then(U, fe);
      }
      K((H = H.apply(T, y || [])).next());
    });
  };
  let s = M(t, "margin", 3, !0);
  const u = In([0, 0]), l = In([0, 0]);
  let c = W(!1);
  function v() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function g() {
    return o(this, void 0, void 0, function* () {
      yield v(), f(c) || g();
    });
  }
  function w() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), g();
    });
  }
  we(() => (w(), () => {
    A(c, !0);
  }));
  var d = Vo();
  let m;
  var P = re(d), h = re(P), b = V(h);
  $(() => {
    m = tt(d, 1, "svelte-m6d381", null, m, { margin: s() }), Oe(h, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Oe(b, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), R(e, d), Yt(), a();
}
var Xo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(c) {
      try {
        l(n.next(c));
      } catch (v) {
        o(v);
      }
    }
    function u(c) {
      try {
        l(n.throw(c));
      } catch (v) {
        o(v);
      }
    }
    function l(c) {
      c.done ? a(c.value) : i(c.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let Dt = [], yr = !1;
const qo = typeof window < "u", xi = qo ? window.requestAnimationFrame : (e) => {
};
function Wo(e) {
  return Xo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Dt.push(t), !yr) yr = !0;
      else return;
      yield va(), xi(() => {
        let n = [0, 0];
        for (let i = 0; i < Dt.length; i++) {
          const o = Dt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), yr = !1, Dt = [];
      });
    }
  });
}
var Zo = /* @__PURE__ */ ue('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), Yo = /* @__PURE__ */ ue('<div class="eta-bar svelte-124hqw6"></div>'), Jo = /* @__PURE__ */ ue("<!> ", 1), Qo = /* @__PURE__ */ ue("<!> <!> <!> <!>", 1), Ko = /* @__PURE__ */ ue('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), $o = /* @__PURE__ */ ue('<p class="loading svelte-124hqw6"> </p> <!>', 1), el = /* @__PURE__ */ ue("<!> <div><!> <!></div> <!> <!>", 1), tl = /* @__PURE__ */ ue('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), rl = /* @__PURE__ */ ue("<div> <!> </div>"), nl = /* @__PURE__ */ ue('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function il(e, t) {
  Jt(t, !0);
  let r = M(t, "eta", 3, null), n = M(t, "scroll_to_output", 3, !1), i = M(t, "timer", 3, !0), a = M(t, "show_progress", 3, "full"), o = M(t, "message", 3, null), s = M(t, "progress", 3, null), u = M(t, "variant", 3, "default"), l = M(t, "loading_text", 3, "Loading..."), c = M(t, "absolute", 3, !0), v = M(t, "translucent", 3, !1), g = M(t, "border", 3, !1), w = M(t, "validation_error", 7, null), d = M(t, "show_validation_error", 3, !0), m = M(t, "type", 3, null), P = M(t, "used_cache", 3, null), h = M(t, "cache_duration", 3, null), b = M(t, "avg_time", 3, null), T, y = !1, x = W(0), H = W(null), O = W(null), B = W(!1), G = W(null), U = W(!1), fe = W(!1), K = W(null), Z = W(null), z = W("from cache"), Be = W(!1), Ie = null, je = null;
  const Ye = Me(() => !(d() && w()) && (m() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Ve = W(0);
  const ke = Me(() => f(O) === null || f(O) <= 0 || !f(Ve) ? 0 : Math.min(f(Ve) / f(O), 1)), ae = Me(() => f(Ve).toFixed(1));
  let Te = Me(() => s() == null), le = Me(() => r() !== null && r() !== void 0 ? r() : f(H));
  function Ne() {
    xi(() => {
      A(Ve, (performance.now() - f(x)) / 1e3), y && Ne();
    });
  }
  let ne = Me(() => {
    let X = null;
    s() != null ? X = s().map((oe) => {
      if (oe.index != null && oe.length != null)
        return oe.index / oe.length;
      if (oe.progress != null)
        return oe.progress;
    }) : X = null;
    let ee, se = "";
    return X ? (ee = X[X.length - 1], ee === 0 ? se = "0" : se = "150ms") : ee = void 0, {
      progress_level: X,
      last_progress_level: ee,
      progress_bar_transition: se
    };
  });
  function _e() {
    y || (A(H, A(G, null), !0), A(x, performance.now(), !0), y = !0, Ne());
  }
  function De() {
    A(H, A(G, null), !0), y && (y = !1);
  }
  we(() => {
    t.status === "pending" ? _e() : ie(() => {
      De();
    });
  }), we(() => {
    T && n() && (t.status === "pending" || t.status === "complete") && Wo(T, t.autoscroll);
  }), we(() => {
    f(le) != null && f(H) !== f(le) && (A(O, (performance.now() - f(x)) / 1e3 + f(le)), A(G, f(O).toFixed(1), !0), A(H, f(le), !0));
  });
  function Se() {
    A(B, !1);
  }
  we(() => {
    ie(() => {
      Se();
    }), t.status === "error" && o() && A(B, !0);
  }), we(() => {
    t.status === "complete" && m() === "output" && P() && h() != null && (A(K, h().toFixed(1), !0), A(z, P() === "full" ? "from cache" : "used cache", !0), A(Be, b() != null && b() > h() && b() > 0, !0), A(Z, f(Be) ? b().toFixed(1) : null, !0), A(U, !0), A(fe, !1), Ie && clearTimeout(Ie), je && clearTimeout(je), Ie = setTimeout(
      () => {
        A(fe, !0), je = setTimeout(
          () => {
            A(U, !1), A(fe, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var ze = nl(), Le = de(ze);
  let Je, ye;
  var pe = re(Le);
  {
    var rr = (X) => {
      var ee = Zo(), se = re(ee), oe = V(se), be = re(oe);
      {
        let Ae = Me(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Pn(be, {
          get Icon() {
            return On;
          },
          get label() {
            return f(Ae);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => w(null)
        });
      }
      $(() => he(se, `${w() ?? ""} `)), R(X, ee);
    };
    Q(pe, (X) => {
      w() && d() && X(rr);
    });
  }
  var Xe = V(pe, 2);
  {
    var nr = (X) => {
      var ee = el(), se = de(ee);
      {
        var oe = (L) => {
          var D = Yo();
          let J;
          $(() => J = Oe(D, "", J, {
            transform: `translateX(${(f(ke) || 0) * 100 - 100}%)`
          })), R(L, D);
        };
        Q(se, (L) => {
          u() === "default" && f(Te) && a() === "full" && L(oe);
        });
      }
      var be = V(se, 2);
      let Ae;
      var p = re(be);
      {
        var _ = (L) => {
          var D = ut(), J = de(D);
          ln(J, 17, s, sn, (xe, ge) => {
            var Nt = ut(), sr = de(Nt);
            {
              var Lt = (Qe) => {
                var _t = Jo(), Ct = de(_t);
                {
                  var or = (Ue) => {
                    var Ke = Re();
                    $((yt, xt) => he(Ke, `${yt ?? ""}/${xt ?? ""}`), [
                      () => _r(f(ge).index || 0),
                      () => _r(f(ge).length)
                    ]), R(Ue, Ke);
                  }, at = (Ue) => {
                    var Ke = Re();
                    $((yt) => he(Ke, yt), [() => _r(f(ge).index || 0)]), R(Ue, Ke);
                  };
                  Q(Ct, (Ue) => {
                    f(ge).length != null ? Ue(or) : Ue(at, -1);
                  });
                }
                var st = V(Ct);
                $(() => he(st, ` ${f(ge).unit ?? ""} |  `)), R(Qe, _t);
              };
              Q(sr, (Qe) => {
                f(ge).index != null && Qe(Lt);
              });
            }
            R(xe, Nt);
          }), R(L, D);
        }, S = (L) => {
          var D = Re();
          $(() => he(D, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), R(L, D);
        }, E = (L) => {
          var D = Re("processing |");
          R(L, D);
        };
        Q(p, (L) => {
          s() ? L(_) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? L(S, 1) : t.queue_position === 0 && L(E, 2);
        });
      }
      var j = V(p, 2);
      {
        var I = (L) => {
          var D = Re();
          $(() => he(D, `${f(ae) ?? ""}${r() ? `/${f(G)}` : ""}s`)), R(L, D);
        };
        Q(j, (L) => {
          i() && L(I);
        });
      }
      var N = V(be, 2);
      {
        var Y = (L) => {
          var D = Ko(), J = re(D), xe = re(J);
          {
            var ge = (Qe) => {
              var _t = ut(), Ct = de(_t);
              ln(Ct, 17, s, sn, (or, at, st) => {
                var Ue = ut(), Ke = de(Ue);
                {
                  var yt = (xt) => {
                    var Yr = Qo(), Jr = de(Yr);
                    {
                      var Ei = (ce) => {
                        var Fe = Re(" /");
                        R(ce, Fe);
                      };
                      Q(Jr, (ce) => {
                        st !== 0 && ce(Ei);
                      });
                    }
                    var Qr = V(Jr, 2);
                    {
                      var wi = (ce) => {
                        var Fe = Re();
                        $(() => he(Fe, f(at).desc)), R(ce, Fe);
                      };
                      Q(Qr, (ce) => {
                        f(at).desc != null && ce(wi);
                      });
                    }
                    var Kr = V(Qr, 2);
                    {
                      var Ti = (ce) => {
                        var Fe = Re("-");
                        R(ce, Fe);
                      };
                      Q(Kr, (ce) => {
                        f(at).desc != null && f(ne).progress_level && f(ne).progress_level[st] != null && ce(Ti);
                      });
                    }
                    var Si = V(Kr, 2);
                    {
                      var Ai = (ce) => {
                        var Fe = Re();
                        $((Hi) => he(Fe, `${Hi ?? ""}%`), [
                          () => (100 * (f(ne).progress_level[st] || 0)).toFixed(1)
                        ]), R(ce, Fe);
                      };
                      Q(Si, (ce) => {
                        f(ne).progress_level != null && ce(Ai);
                      });
                    }
                    R(xt, Yr);
                  };
                  Q(Ke, (xt) => {
                    (f(at).desc != null || f(ne).progress_level && f(ne).progress_level[st] != null) && xt(yt);
                  });
                }
                R(or, Ue);
              }), R(Qe, _t);
            };
            Q(xe, (Qe) => {
              s() != null && Qe(ge);
            });
          }
          var Nt = V(J, 2), sr = re(Nt);
          let Lt;
          $(() => Lt = Oe(sr, "", Lt, {
            width: `${f(ne).last_progress_level * 100}%`,
            transition: f(ne).progress_bar_transition
          })), R(L, D);
        }, te = (L) => {
          {
            let D = Me(() => u() === "default");
            zo(L, {
              get margin() {
                return f(D);
              }
            });
          }
        };
        Q(N, (L) => {
          f(ne).last_progress_level != null ? L(Y) : a() === "full" && L(te, 1);
        });
      }
      var Ce = V(N, 2);
      {
        var ve = (L) => {
          var D = $o(), J = de(D), xe = re(J), ge = V(J, 2);
          Ar(ge, t, "additional-loading-text", {}), $(() => he(xe, l())), R(L, D);
        };
        Q(Ce, (L) => {
          i() || L(ve);
        });
      }
      $(() => Ae = tt(be, 1, "progress-text svelte-124hqw6", null, Ae, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), R(X, ee);
    }, It = (X) => {
      var ee = tl(), se = de(ee), oe = re(se);
      {
        let _ = Me(() => t.i18n("common.clear"));
        Pn(oe, {
          get Icon() {
            return On;
          },
          get label() {
            return f(_);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var be = V(se, 2), Ae = re(be), p = V(be, 2);
      Ar(p, t, "error", {}), $((_) => he(Ae, _), [() => t.i18n("common.error")]), R(X, ee);
    };
    Q(Xe, (X) => {
      t.status === "pending" ? X(nr) : t.status === "error" && X(It, 1);
    });
  }
  qr(Le, (X) => T = X, () => T);
  var ir = V(Le, 2);
  {
    var ar = (X) => {
      var ee = rl();
      let se, oe;
      var be = re(ee), Ae = V(be);
      {
        var p = (S) => {
          var E = Re();
          $(() => he(E, `~${f(Z) ?? ""}s
			→ `)), R(S, E);
        };
        Q(Ae, (S) => {
          f(Be) && S(p);
        });
      }
      var _ = V(Ae);
      $(() => {
        se = tt(ee, 1, "cache-indicator svelte-124hqw6", null, se, { "fade-out": f(fe) }), oe = Oe(ee, "", oe, { position: c() ? "absolute" : "static" }), he(be, `⚡ ${f(z) ?? ""}: `), he(_, `${f(K) ?? ""}s`);
      }), R(X, ee);
    };
    Q(ir, (X) => {
      f(U) && X(ar);
    });
  }
  $(() => {
    Je = tt(Le, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", Je, {
      "no-click": w() && d(),
      hide: f(Ye),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || v() || a() === "minimal" || w(),
      generating: t.status === "generating" && a() === "full",
      border: g()
    }), ye = Oe(Le, "", ye, {
      position: c() ? "absolute" : "static",
      padding: c() ? "0" : "var(--size-8) 0"
    });
  }), R(e, ze), Yt();
}
const al = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, sl = [
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
], ol = [
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
], ll = [
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
al([
  Object.fromEntries(sl.map((e) => [e, ["*"]])),
  Object.fromEntries(ol.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(ll.map((e) => [e, ["math:*"]]))
]);
Zt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var ul = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), fl = /* @__PURE__ */ ue('<!> <div class="layout-editor svelte-r41nsf"><div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">重置</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回视野</button></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function hl(e, t) {
  Jt(t, !0);
  const r = /* @__PURE__ */ Ka(t, ul), n = 2048, i = 35e-5, a = new Ro(r);
  let o, s = null, u = null, l = null, c = null, v = W(!1), g = W(!1), w = W("等待版图 mask"), d = W(Ht({ enabled: !1 })), m = W(Ht(O())), P = W(!1), h = W(!1), b = W("crosshair"), T = "", y = { x: 0, y: 0, center_x: 0, center_y: 0 }, x = { angle: 0, rotation: 0 }, H = null;
  function O() {
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
  function B(p) {
    return JSON.parse(JSON.stringify(p || { enabled: !1 }));
  }
  function G(p) {
    return typeof p == "number" ? `${p}px` : p || "520px";
  }
  function U(p, _, S) {
    return Math.max(_, Math.min(S, p));
  }
  function fe(p) {
    let _ = ((p + 180) % 360 + 360) % 360 - 180;
    return _ === -180 && (_ = 180), _;
  }
  function K() {
    return Math.max(1, Number(f(d).target_width || s?.naturalWidth || 1));
  }
  function Z() {
    return Math.max(1, Number(f(d).target_height || s?.naturalHeight || 1));
  }
  function z() {
    return Math.max(1, Number(f(d).source_width || u?.naturalWidth || 1));
  }
  function Be() {
    return Math.max(1, Number(f(d).source_height || u?.naturalHeight || 1));
  }
  function Ie() {
    return Math.min(1, n / Math.max(K(), Z()));
  }
  function je() {
    const p = f(d).foreground_bbox_xyxy;
    return Array.isArray(p) && p.length >= 4 ? p.map(Number) : [0, 0, z() - 1, Be() - 1];
  }
  function Ye(p, _) {
    if (!p) {
      _(null);
      return;
    }
    const S = new Image();
    S.onload = () => _(S), S.onerror = () => _(null), S.src = p;
  }
  function Ve() {
    if (!u) {
      l = null, c = null;
      return;
    }
    const p = z(), _ = Be();
    l = document.createElement("canvas"), l.width = p, l.height = _;
    const S = l.getContext("2d", { willReadFrequently: !0 });
    if (!S) return;
    S.imageSmoothingEnabled = !1, S.drawImage(u, 0, 0, p, _);
    const E = S.getImageData(0, 0, p, _);
    c = document.createElement("canvas"), c.width = p, c.height = _;
    const j = c.getContext("2d");
    if (!j) return;
    const I = j.createImageData(p, _);
    for (let N = 0; N < E.data.length; N += 4) {
      const Y = Math.max(E.data[N], E.data[N + 1], E.data[N + 2]);
      E.data[N + 3] > 0 && Y >= 128 && (I.data[N] = 0, I.data[N + 1] = 255, I.data[N + 2] = 120, I.data[N + 3] = 255);
    }
    j.putImageData(I, 0, 0);
  }
  function ke() {
    H && (clearTimeout(H), H = null);
  }
  function ae(p) {
    ke(), A(d, B(p), !0), A(m, Object.assign(Object.assign({}, O()), f(d).transform || {}), !0), A(w, f(d).status || "编辑器已加载", !0), A(v, !1), A(g, !1), Ye(f(d).base_image, (_) => {
      s = _, A(v, !!_), Xe();
    }), Ye(f(d).mask_image, (_) => {
      u = _, A(g, !!_), Ve(), Xe();
    });
  }
  we(() => {
    const p = JSON.stringify(a.props.value || null);
    p !== T && (T = p, ae(a.props.value));
  }), Pa(() => {
    ke();
  });
  function Te(p) {
    const _ = Number(p.rotation_deg || 0) * Math.PI / 180, S = Number(p.scale || 1), E = Math.cos(_), j = Math.sin(_), I = S * E, N = S * j, Y = Number(p.center_x || 0) - I * Number(p.pivot_x || 0) + N * Number(p.pivot_y || 0), te = Number(p.center_y || 0) - N * Number(p.pivot_x || 0) - I * Number(p.pivot_y || 0);
    return [I, N, -N, I, Y, te];
  }
  function le(p, _, S = f(m)) {
    const [E, j, I, N, Y, te] = Te(S);
    return { x: E * p + I * _ + Y, y: j * p + N * _ + te };
  }
  function Ne(p, _, S = f(m)) {
    const [E, j, I, N, Y, te] = Te(S), Ce = E * N - j * I;
    if (Math.abs(Ce) < 1e-9) return { x: -1, y: -1 };
    const ve = p - Y, L = _ - te;
    return { x: (N * ve - I * L) / Ce, y: (-j * ve + E * L) / Ce };
  }
  function ne(p) {
    const _ = o.getBoundingClientRect();
    return {
      x: (p.clientX - _.left) / Math.max(1, _.width) * K(),
      y: (p.clientY - _.top) / Math.max(1, _.height) * Z()
    };
  }
  function _e(p, _) {
    if (!l) return !1;
    const S = Math.round(p), E = Math.round(_);
    if (S < 0 || E < 0 || S >= l.width || E >= l.height) return !1;
    const j = l.getContext("2d", { willReadFrequently: !0 });
    if (!j) return !1;
    const I = j.getImageData(S, E, 1, 1).data;
    return I[3] > 0 && Math.max(I[0], I[1], I[2]) >= 128;
  }
  function De(p, _) {
    const S = Ne(p, _);
    return _e(S.x, S.y);
  }
  function Se() {
    const [p, _, S, E] = je();
    return [
      le(p, _),
      le(S, _),
      le(S, E),
      le(p, E)
    ];
  }
  function ze() {
    const p = Se();
    let _ = p[0];
    for (const I of p)
      (I.y < _.y || Math.abs(I.y - _.y) < 1e-6 && I.x > _.x) && (_ = I);
    const S = Number(f(m).rotation_deg || 0) * Math.PI / 180, E = Math.cos(S - Math.PI / 4), j = Math.sin(S - Math.PI / 4);
    return { x: _.x + E * 34, y: _.y + j * 34 };
  }
  function Le() {
    const p = o?.getBoundingClientRect();
    return p ? Math.max(8, 14 * K() / Math.max(1, p.width)) : 14;
  }
  function Je(p, _) {
    const S = ze(), E = Le();
    return Math.hypot(p - S.x, _ - S.y) <= E;
  }
  function ye(p) {
    A(
      m,
      Object.assign(Object.assign({}, f(m)), {
        revision: Number(f(m).revision || 0) + 1,
        origin: p,
        scale: U(Number(f(m).scale || 1), 0.01, 20),
        rotation_deg: fe(Number(f(m).rotation_deg || 0))
      }),
      !0
    );
  }
  function pe(p, _, S = !0) {
    ye(p), A(
      d,
      Object.assign(Object.assign({}, f(d)), {
        enabled: !0,
        transform: Object.assign({}, f(m)),
        status: _
      }),
      !0
    ), A(w, _, !0), a.props.value = f(d), T = JSON.stringify(f(d)), S && (ke(), a.dispatch("change")), Xe();
  }
  function rr(p, _, S = 140) {
    pe(p, _, !1), ke(), H = setTimeout(
      () => {
        H = null, a.dispatch("change");
      },
      S
    );
  }
  function Xe() {
    if (!o) return;
    const p = K(), _ = Z(), S = Ie();
    o.width = Math.max(1, Math.round(p * S)), o.height = Math.max(1, Math.round(_ * S));
    const E = o.getContext("2d");
    if (E && (E.setTransform(S, 0, 0, S, 0, 0), E.clearRect(0, 0, p, _), s && f(v) ? (E.imageSmoothingEnabled = !0, E.drawImage(s, 0, 0, p, _)) : (E.fillStyle = "#f8fafc", E.fillRect(0, 0, p, _), E.fillStyle = "#64748b", E.font = "18px sans-serif", E.fillText("请先上传图像", 24, 42)), c && f(g) && f(d).enabled !== !1)) {
      E.save(), E.globalAlpha = U(Number(f(m).preview_alpha || 0.35), 0, 1);
      const [j, I, N, Y, te, Ce] = Te(f(m));
      E.setTransform(S * j, S * I, S * N, S * Y, S * te, S * Ce), E.imageSmoothingEnabled = !1, E.drawImage(c, 0, 0, z(), Be()), E.restore();
      const ve = Se();
      E.save(), E.lineJoin = "round", E.lineWidth = Math.max(2.5, p / 700), E.strokeStyle = "rgba(0,0,0,0.82)", E.beginPath(), E.moveTo(ve[0].x, ve[0].y);
      for (let J = 1; J < ve.length; J++) E.lineTo(ve[J].x, ve[J].y);
      E.closePath(), E.stroke(), E.lineWidth = Math.max(1.8, p / 1e3), E.strokeStyle = "#00ff66", E.stroke();
      const L = ze(), D = ve.reduce((J, xe) => xe.y < J.y || Math.abs(xe.y - J.y) < 1e-6 && xe.x > J.x ? xe : J, ve[0]);
      E.strokeStyle = "#0f172a", E.lineWidth = Math.max(2, p / 900), E.beginPath(), E.moveTo(D.x, D.y), E.lineTo(L.x, L.y), E.stroke(), E.fillStyle = f(h) ? "#ffb000" : "#ffffff", E.strokeStyle = "#00ff66", E.lineWidth = Math.max(2, p / 900), E.beginPath(), E.arc(L.x, L.y, Le(), 0, Math.PI * 2), E.fill(), E.stroke(), E.restore();
    }
  }
  function nr(p) {
    if (p.button !== 0) return;
    if (!f(d).enabled || !f(g) || !l) {
      A(w, "请先启用并加载版图 mask"), Xe();
      return;
    }
    ke();
    const _ = ne(p);
    if (Je(_.x, _.y)) {
      A(b, "grabbing"), A(h, !0), x = {
        angle: Math.atan2(_.y - f(m).center_y, _.x - f(m).center_x) * 180 / Math.PI,
        rotation: Number(f(m).rotation_deg || 0)
      }, o.setPointerCapture(p.pointerId);
      return;
    }
    if (!De(_.x, _.y)) {
      A(w, "请点中版图 mask 前景后拖动"), Xe();
      return;
    }
    A(b, "grabbing"), A(P, !0), y = {
      x: _.x,
      y: _.y,
      center_x: Number(f(m).center_x || 0),
      center_y: Number(f(m).center_y || 0)
    }, o.setPointerCapture(p.pointerId);
  }
  function It(p) {
    if (!f(d).enabled || !f(g) || !l) {
      A(b, "not-allowed");
      return;
    }
    if (Je(p.x, p.y) || De(p.x, p.y)) {
      A(b, "grab");
      return;
    }
    A(b, "crosshair");
  }
  function ir(p) {
    const _ = ne(p);
    if (!f(P) && !f(h)) {
      It(_);
      return;
    }
    if (A(b, "grabbing"), f(P))
      A(
        m,
        Object.assign(Object.assign({}, f(m)), {
          center_x: y.center_x + _.x - y.x,
          center_y: y.center_y + _.y - y.y
        }),
        !0
      ), A(w, "正在拖动；松开后同步变换");
    else if (f(h)) {
      const S = Math.atan2(_.y - f(m).center_y, _.x - f(m).center_x) * 180 / Math.PI;
      A(
        m,
        Object.assign(Object.assign({}, f(m)), {
          rotation_deg: fe(x.rotation + S - x.angle)
        }),
        !0
      ), A(w, "正在旋转；松开后同步变换");
    }
    Xe();
  }
  function ar() {
    !f(P) && !f(h) && A(b, "crosshair");
  }
  function X(p) {
    if (f(P) || f(h)) {
      A(P, !1), A(h, !1);
      try {
        o.releasePointerCapture(p.pointerId);
      } catch {
      }
      It(ne(p)), pe("canvas", "Canvas 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function ee(p) {
    if (!f(d).enabled || !f(g)) return;
    p.preventDefault();
    const _ = ne(p), S = Ne(_.x, _.y), E = Math.exp(-p.deltaY * i), j = U(Number(f(m).scale || 1) * E, 0.01, 20);
    let I = Object.assign(Object.assign({}, f(m)), { scale: j });
    const N = le(S.x, S.y, I);
    I = Object.assign(Object.assign({}, I), {
      center_x: Number(I.center_x || 0) + _.x - N.x,
      center_y: Number(I.center_y || 0) + _.y - N.y
    }), A(m, I, !0), rr("canvas", `滚轮缩放已同步: scale=${j.toFixed(3)}`);
  }
  function se() {
    A(
      m,
      Object.assign(Object.assign({}, f(m)), {
        center_x: K() / 2,
        center_y: Z() / 2,
        scale: 1,
        rotation_deg: 0
      }),
      !0
    ), pe("reset", "重置：已居中，scale=1，rotation=0");
  }
  function oe() {
    A(m, Object.assign(Object.assign({}, f(m)), { center_x: K() / 2, center_y: Z() / 2 }), !0), pe("center", "居中：保留缩放和旋转");
  }
  function be() {
    const [p, _, S, E] = je(), j = Math.max(1, S - p + 1), I = Math.max(1, E - _ + 1), N = U(Math.min(K() / j, Z() / I) * 0.9, 0.01, 20);
    A(
      m,
      Object.assign(Object.assign({}, f(m)), {
        center_x: K() / 2,
        center_y: Z() / 2,
        scale: N
      }),
      !0
    ), pe("fit", `适配：scale=${N.toFixed(3)}`);
  }
  function Ae() {
    const p = Se(), _ = Math.min(...p.map((te) => te.x)), S = Math.max(...p.map((te) => te.x)), E = Math.min(...p.map((te) => te.y)), j = Math.max(...p.map((te) => te.y));
    let I = 0, N = 0;
    const Y = Math.max(20, K() * 0.03);
    if (S < Y ? I = Y - S : _ > K() - Y && (I = K() - Y - _), j < Y ? N = Y - j : E > Z() - Y && (N = Z() - Y - E), I === 0 && N === 0) {
      A(w, "版图已经在视野内");
      return;
    }
    A(
      m,
      Object.assign(Object.assign({}, f(m)), {
        center_x: Number(f(m).center_x || 0) + I,
        center_y: Number(f(m).center_y || 0) + N
      }),
      !0
    ), pe("bring_into_view", "已找回到视野内");
  }
  {
    let p = Me(() => f(P) || f(h) ? "focus" : "base");
    Uo(e, {
      get visible() {
        return a.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return f(p);
      },
      padding: !1,
      get elem_id() {
        return a.shared.elem_id;
      },
      get elem_classes() {
        return a.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return a.shared.container;
      },
      get scale() {
        return a.shared.scale;
      },
      get min_width() {
        return a.shared.min_width;
      },
      children: (_, S) => {
        var E = fl(), j = de(E);
        il(j, es(
          {
            get autoscroll() {
              return a.shared.autoscroll;
            },
            get i18n() {
              return a.i18n;
            }
          },
          () => a.shared.loading_status,
          {
            on_clear_status: () => a.dispatch("clear_status", a.shared.loading_status)
          }
        ));
        var I = V(j, 2), N = re(I), Y = re(N), te = V(Y, 2), Ce = V(te, 2), ve = V(Ce, 2), L = V(N, 2), D = re(L);
        qr(D, (ge) => o = ge, () => o);
        var J = V(L, 2), xe = re(J);
        $(
          (ge) => {
            Oe(I, ge), Oe(D, `cursor:${f(b)}`), he(xe, f(w));
          },
          [() => `min-height:${G(a.props.height)}`]
        ), He("click", Y, se), He("click", te, oe), He("click", Ce, be), He("click", ve, Ae), He("pointerdown", D, nr), He("pointermove", D, ir), He("pointerup", D, X), He("pointercancel", D, X), He("pointerleave", D, ar), He("wheel", D, ee), R(_, E);
      },
      $$slots: { default: !0 }
    });
  }
  Yt();
}
export {
  hl as default
};
