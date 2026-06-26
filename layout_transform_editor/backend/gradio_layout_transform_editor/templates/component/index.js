import { i as Dr, g as On, o as Mi, n as We, u as ne, s as Pi, r as xr, m as Je, a as A, b as f, t as Ur, d as Bi, q as Ii, c as Cn, e as $e, f as Wt, h as Vt, j as Ni, T as Oi, k as Ci, l as zt, p as Qe, v as Fr, w as et, x as Ln, y as Rn, z as kn, A as Pt, E as Zt, B as ut, C as Dn, D as Te, F as $r, G as Li, H as Un, I as Gr, J as Ri, K as en, L as ki, M as Di, N as Fe, O as Fn, P as lr, Q as Ui, R as Fi, S as Gi, U as ji, V as Gn, W as jr, X as tn, Y as rn, Z as Vi, _ as zi, $ as Xi, a0 as qi, a1 as Wi, a2 as Zi, a3 as Yi, a4 as Ji, a5 as Vr, a6 as Qi, a7 as jn, a8 as Yt, a9 as Ki, aa as $i, ab as ea, ac as ta, ad as ra, ae as na, af as zr, ag as ia, ah as aa, ai as we, aj as Er, ak as wr, al as sa, am as oa, an as Ht, ao as la, ap as ua, aq as fa, ar as ca, as as ha, at as Vn, au as Et, av as Y, aw as da, ax as nn, ay as ma, az as de, aA as Jt, aB as Qt, aC as V, aD as Ae, aE as Q, aF as pa, aG as ee, aH as he, aI as He, aJ as va } from "./render-BGdAxg9I.js";
function zn(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const ga = [];
function ba(e, t = !1, r = !1) {
  return Ft(e, /* @__PURE__ */ new Map(), "", ga, null, r);
}
function Ft(e, t, r, n, i = null, a = !1) {
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
        u in e && (s[u] = Ft(l, t, r, n, null, a));
      }
      return s;
    }
    if (On(e) === Mi) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var c of Object.keys(e))
        s[c] = Ft(
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
      return Ft(
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
    return t(void 0), r && r(void 0), We;
  const n = ne(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const st = [];
function _a(e, t) {
  return {
    subscribe: Bt(e, t).subscribe
  };
}
function Bt(e, t = We) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Pi(e, s) && (e = s, r)) {
      const u = !st.length;
      for (const l of n)
        l[1](), st.push(l, e);
      if (u) {
        for (let l = 0; l < st.length; l += 2)
          st[l][0](st[l + 1]);
        st.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = We) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || We), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function pt(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return _a(r, (o, s) => {
    let u = !1;
    const l = [];
    let c = 0, p = We;
    const v = () => {
      if (c)
        return;
      p();
      const h = t(n ? l[0] : l, o, s);
      a ? o(h) : p = typeof h == "function" ? h : We;
    }, x = i.map(
      (h, w) => Xr(
        h,
        (H) => {
          l[w] = H, c &= ~(1 << w), u && v();
        },
        () => {
          c |= 1 << w;
        }
      )
    );
    return u = !0, v(), function() {
      xr(x), p(), u = !1;
    };
  });
}
function ya(e) {
  let t;
  return Xr(e, (r) => t = r)(), t;
}
let kt = !1, Tr = /* @__PURE__ */ Symbol("unmounted");
function an(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Je(void 0),
    unsubscribe: We
  };
  if (n.store !== e && !(Tr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = We;
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
      Bi(e, Tr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = kt;
  try {
    return kt = !1, [e(), kt];
  } finally {
    kt = t;
  }
}
function wa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Ii(() => {
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
  var t = Cn("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function ct(e, t) {
  var r = (
    /** @type {Effect} */
    Wt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function fe(e, t) {
  var r = (t & Oi) !== 0, n = (t & Ci) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Xn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    Vt(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ni ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        Vt(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      ct(s, u);
    } else
      ct(o, o);
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
        Vt(o)
      );
      a = /** @type {Element} */
      Vt(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return ct(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function qn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function Le(e = "") {
  {
    var t = $e(e + "");
    return ct(t, t), t;
  }
}
function lt() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = $e();
  return e.append(t, r), ct(t, r), e;
}
function R(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Kt {
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
        zt(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (zt(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (Qe(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Rn(o, l), l.append($e()), this.#e.set(a, { effect: o, fragment: l });
          } else
            Qe(o);
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
      r.includes(n) || (Qe(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Ln
    ), i = kn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = $e();
        a.append(o), this.#e.set(t, {
          effect: et(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          et(() => r(this.anchor))
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
  var n = new Kt(e);
  Pt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Zt);
}
function Ma(e) {
  ut === null && zn(), Dn && ut.l !== null ? Ba(ut).m.push(e) : Te(() => {
    const t = ne(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Pa(e) {
  ut === null && zn(), Ma(() => () => ne(e));
}
function Ba(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function J(e, t, r = !1) {
  var n = new Kt(e), i = r ? Zt : 0;
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
function Ia(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let p = t[s];
    Fr(
      p,
      () => {
        if (a) {
          if (a.pending.delete(p), a.done.add(p), a.pending.size === 0) {
            var v = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Sr(e, Gr(a.done)), v.delete(a), v.size === 0 && (e.outrogroups = null);
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
      a.f |= Fe;
      const o = document.createDocumentFragment();
      Rn(a, o);
    } else
      Qe(t[i], r);
  }
}
var on;
function ln(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = Un(() => {
    var d = r();
    return (
      /** @type {V[]} */
      Dr(d) ? d : d == null ? [] : Gr(d)
    );
  }), c, p = /* @__PURE__ */ new Map(), v = !0;
  function x(d) {
    (H.effect.f & Fn) === 0 && (H.pending.delete(d), H.fallback = u, Na(H, c, o, t, n), u !== null && (c.length === 0 ? (u.f & Fe) === 0 ? zt(u) : (u.f ^= Fe, St(u, null, o)) : Fr(u, () => {
      u = null;
    })));
  }
  function h(d) {
    H.pending.delete(d);
  }
  var w = Pt(() => {
    c = /** @type {V[]} */
    f(l);
    for (var d = c.length, g = /* @__PURE__ */ new Set(), T = (
      /** @type {Batch} */
      Ln
    ), y = kn(), _ = 0; _ < d; _ += 1) {
      var P = c[_], B = n(P, _), I = v ? null : s.get(B);
      I ? (I.v && $r(I.v, P), I.i && $r(I.i, _), y && T.unskip_effect(I.e)) : (I = Oa(
        s,
        v ? o : on ??= $e(),
        P,
        B,
        _,
        i,
        t,
        r
      ), v || (I.e.f |= Fe), s.set(B, I)), g.add(B);
    }
    if (d === 0 && a && !u && (v ? u = et(() => a(o)) : (u = et(() => a(on ??= $e())), u.f |= Fe)), d > g.size && Li(), !v)
      if (p.set(T, g), y) {
        for (const [k, j] of s)
          g.has(k) || T.skip_effect(j.e);
        T.oncommit(x), T.ondiscard(h);
      } else
        x(T);
    f(l);
  }), H = { effect: w, items: s, pending: p, outrogroups: null, fallback: u };
  v = !1;
}
function wt(e) {
  for (; e !== null && (e.f & Ui) === 0; )
    e = e.next;
  return e;
}
function Na(e, t, r, n, i) {
  var a = t.length, o = e.items, s = wt(e.effect.first), u, l = null, c = [], p = [], v, x, h, w;
  for (w = 0; w < a; w += 1) {
    if (v = t[w], x = i(v, w), h = /** @type {EachItem} */
    o.get(x).e, e.outrogroups !== null)
      for (const I of e.outrogroups)
        I.pending.delete(h), I.done.delete(h);
    if ((h.f & lr) !== 0 && zt(h), (h.f & Fe) !== 0)
      if (h.f ^= Fe, h === s)
        St(h, null, r);
      else {
        var H = l ? l.next : s;
        h === e.effect.last && (e.effect.last = h.prev), h.prev && (h.prev.next = h.next), h.next && (h.next.prev = h.prev), Xe(e, l, h), Xe(e, h, H), St(h, H, r), l = h, c = [], p = [], s = wt(l.next);
        continue;
      }
    if (h !== s) {
      if (u !== void 0 && u.has(h)) {
        if (c.length < p.length) {
          var d = p[0], g;
          l = d.prev;
          var T = c[0], y = c[c.length - 1];
          for (g = 0; g < c.length; g += 1)
            St(c[g], d, r);
          for (g = 0; g < p.length; g += 1)
            u.delete(p[g]);
          Xe(e, T.prev, y.next), Xe(e, l, T), Xe(e, y, d), s = d, l = y, w -= 1, c = [], p = [];
        } else
          u.delete(h), St(h, s, r), Xe(e, h.prev, h.next), Xe(e, h, l === null ? e.effect.first : l.next), Xe(e, l, h), l = h;
        continue;
      }
      for (c = [], p = []; s !== null && s !== h; )
        (u ??= /* @__PURE__ */ new Set()).add(s), p.push(s), s = wt(s.next);
      if (s === null)
        continue;
    }
    (h.f & Fe) === 0 && c.push(h), l = h, s = wt(h.next);
  }
  if (e.outrogroups !== null) {
    for (const I of e.outrogroups)
      I.pending.size === 0 && (Sr(e, Gr(I.done)), e.outrogroups?.delete(I));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var _ = [];
    if (u !== void 0)
      for (h of u)
        (h.f & lr) === 0 && _.push(h);
    for (; s !== null; )
      (s.f & lr) === 0 && s !== e.fallback && _.push(s), s = wt(s.next);
    var P = _.length;
    if (P > 0) {
      var B = null;
      Ia(e, _, B);
    }
  }
}
function Oa(e, t, r, n, i, a, o, s) {
  var u = (o & ki) !== 0 ? (o & Di) === 0 ? Je(r, !1, !1) : en(r) : null, l = (o & Ri) !== 0 ? en(i) : null;
  return {
    v: u,
    i: l,
    e: et(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function St(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Fe) === 0 ? (
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
function Xe(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Ar(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Ca(e, t, r) {
  var n = new Kt(e);
  Pt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Zt);
}
const La = () => performance.now(), Me = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => La(),
  tasks: /* @__PURE__ */ new Set()
};
function Wn() {
  const e = Me.now();
  Me.tasks.forEach((t) => {
    t.c(e) || (Me.tasks.delete(t), t.f());
  }), Me.tasks.size !== 0 && Me.tick(Wn);
}
function Ra(e) {
  let t;
  return Me.tasks.size === 0 && Me.tick(Wn), {
    promise: new Promise((r) => {
      Me.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Me.tasks.delete(t);
    }
  };
}
function ka(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new Kt(s, !1);
  Pt(() => {
    const l = t() || null;
    var c = l === "svg" ? ji : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (p) => {
      if (l) {
        if (o = Cn(l, c), ct(o, o), n) {
          var v = null, x = o.appendChild($e());
          n(o, x), v?.remove();
        }
        Wt.nodes.end = o, p.before(o);
      }
    }), () => {
    };
  }, Zt), Ur(() => {
  });
}
function Da(e, t) {
  var r = void 0, n;
  Gn(() => {
    r !== (r = t()) && (n && (Qe(n), n = null), r && (n = et(() => {
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
      const w = e.length;
      for (var p = 0; p < w; p++) {
        var v = e[p];
        if (s ? v === "/" && e[p - 1] === "*" && (s = !1) : a ? a === v && (a = !1) : v === "/" && e[p + 1] === "*" ? s = !0 : v === '"' || v === "'" ? a = v : v === "(" ? o++ : v === ")" && o--, !s && a === !1 && o === 0) {
          if (v === ":" && c === -1)
            c = p;
          else if (v === ";" || p === w - 1) {
            if (c !== -1) {
              var x = ur(e.substring(l, c).trim());
              if (!u.includes(x)) {
                v !== ";" && p++;
                var h = e.substring(l, p).trim();
                r += " " + h + ";";
              }
            }
            l = p + 1, c = -1;
          }
        }
      }
    }
    return n && (r += fn(n)), i && (r += fn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Ke(e, t, r, n, i, a) {
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
function Pe(e, t, r, n) {
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
const At = /* @__PURE__ */ Symbol("class"), ot = /* @__PURE__ */ Symbol("style"), Yn = /* @__PURE__ */ Symbol("is custom element"), Jn = /* @__PURE__ */ Symbol("is html"), za = Vr ? "input" : "INPUT", Xa = Vr ? "option" : "OPTION", qa = Vr ? "select" : "SELECT";
function Wa(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function ft(e, t, r, n) {
  var i = Qn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Xi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Kn(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Za(e, t, r, n, i = !1, a = !1) {
  var o = Qn(e), s = o[Yn], u = !o[Jn], l = t || {}, c = e.nodeName === Xa;
  for (var p in t)
    p in r || (r[p] = null);
  r.class ? r.class = Fa(r.class) : r.class = null, r[ot] && (r.style ??= null);
  var v = Kn(e);
  if (e.nodeName === za && "type" in r && ("value" in r || "__value" in r)) {
    var x = r.type;
    (x !== l.type || x === void 0 && e.hasAttribute("type")) && (l.type = x, ft(e, "type", x));
  }
  for (const y in r) {
    let _ = r[y];
    if (c && y === "value" && _ == null) {
      e.value = e.__value = "", l[y] = _;
      continue;
    }
    if (y === "class") {
      var h = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Ke(e, h, _, n, t?.[At], r[At]), l[y] = _, l[At] = r[At];
      continue;
    }
    if (y === "style") {
      Pe(e, _, t?.[ot], r[ot]), l[y] = _, l[ot] = r[ot];
      continue;
    }
    var w = l[y];
    if (!(_ === w && !(_ === void 0 && e.hasAttribute(y)))) {
      l[y] = _;
      var H = y[0] + y[1];
      if (H !== "$$")
        if (H === "on") {
          const P = {}, B = "$$" + y;
          let I = y.slice(2);
          var d = ta(I);
          if (Qi(I) && (I = I.slice(0, -7), P.capture = !0), !d && w) {
            if (_ != null) continue;
            e.removeEventListener(I, l[B], P), l[B] = null;
          }
          if (d)
            jn(I, e, _), Yt([I]);
          else if (_ != null) {
            let k = function(j) {
              l[y].call(this, j);
            };
            l[B] = Ki(I, e, k, P);
          }
        } else if (y === "style")
          ft(e, y, _);
        else if (y === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!_
          );
        else if (!s && (y === "__value" || y === "value" && _ != null))
          e.value = e.__value = _;
        else if (y === "selected" && c)
          Wa(
            /** @type {HTMLOptionElement} */
            e,
            _
          );
        else {
          var g = y;
          u || (g = $i(g));
          var T = g === "defaultValue" || g === "defaultChecked";
          if (_ == null && !s && !T)
            if (o[y] = null, g === "value" || g === "checked") {
              let P = (
                /** @type {HTMLInputElement} */
                e
              );
              const B = t === void 0;
              if (g === "value") {
                let I = P.defaultValue;
                P.removeAttribute(g), P.defaultValue = I, P.value = P.__value = B ? I : null;
              } else {
                let I = P.defaultChecked;
                P.removeAttribute(g), P.defaultChecked = I, P.checked = B ? I : !1;
              }
            } else
              e.removeAttribute(y);
          else T || v.includes(g) && (s || typeof _ != "string") ? (e[g] = _, g in o && (o[g] = ea)) : typeof _ != "function" && ft(e, g, _);
        }
    }
  }
  return l;
}
function Ya(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Yi(i, r, n, (u) => {
    var l = void 0, c = {}, p = e.nodeName === qa, v = !1;
    if (Gn(() => {
      var h = t(...u.map(f)), w = Za(
        e,
        l,
        h,
        a,
        o,
        s
      );
      v && p && "value" in h && Hr(
        /** @type {HTMLSelectElement} */
        e,
        h.value
      );
      for (let d of Object.getOwnPropertySymbols(c))
        h[d] || Qe(c[d]);
      for (let d of Object.getOwnPropertySymbols(h)) {
        var H = h[d];
        d.description === Ji && (!l || H !== l[d]) && (c[d] && Qe(c[d]), c[d] = et(() => Da(e, () => H))), w[d] = H;
      }
      l = w;
    }), p) {
      var x = (
        /** @type {HTMLSelectElement} */
        e
      );
      jr(() => {
        Hr(
          x,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), Va(x);
      });
    }
    v = !0;
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
    i = On(i);
  }
  return r;
}
function cr(e, t) {
  return e === t || e?.[zr] === t;
}
function qr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    ut.r
  ), a = (
    /** @type {Effect} */
    Wt
  );
  return jr(() => {
    var o, s;
    return ra(() => {
      o = s, s = [], ne(() => {
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
    ut
  ), r = t.l.u;
  if (!r) return;
  let n = () => we(t.s);
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
  }), Te(() => {
    const i = ne(() => r.m.map(aa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Te(() => {
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
  ), f(l)) : (u && (u = !1, s = o ? ne(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let p;
  if (a) {
    var v = zr in e || Vn in e;
    p = wr(e, t)?.set ?? (v && t in e ? (y) => e[t] = y : void 0);
  }
  var x, h = !1;
  a ? [x, h] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : x = /** @type {V} */
  e[t], x === void 0 && n !== void 0 && (x = c(), p && (i && sa(), p(x)));
  var w;
  if (i ? w = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y === void 0 ? c() : (u = !0, y);
  } : w = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y !== void 0 && (s = /** @type {V} */
    void 0), y === void 0 ? s : y;
  }, i && (r & oa) === 0)
    return w;
  if (p) {
    var H = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(y, _) {
        return arguments.length > 0 ? ((!i || !_ || H || h) && p(_ ? w() : y), y) : w();
      })
    );
  }
  var d = !1, g = ((r & fa) !== 0 ? Er : Un)(() => (d = !1, w()));
  a && f(g);
  var T = (
    /** @type {Effect} */
    Wt
  );
  return (
    /** @type {() => V} */
    (function(y, _) {
      if (arguments.length > 0) {
        const P = _ ? f(g) : i && a ? Ht(y) : y;
        return A(g, P), d = !0, s !== void 0 && (s = P), y;
      }
      return ha && d || (T.f & Fn) !== 0 ? g.v : f(g);
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
  var e = function(g) {
    return t(g) && !r(g);
  };
  function t(d) {
    return !!d && typeof d == "object";
  }
  function r(d) {
    var g = Object.prototype.toString.call(d);
    return g === "[object RegExp]" || g === "[object Date]" || a(d);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(d) {
    return d.$$typeof === i;
  }
  function o(d) {
    return Array.isArray(d) ? [] : {};
  }
  function s(d, g) {
    return g.clone !== !1 && g.isMergeableObject(d) ? w(o(d), d, g) : d;
  }
  function u(d, g, T) {
    return d.concat(g).map(function(y) {
      return s(y, T);
    });
  }
  function l(d, g) {
    if (!g.customMerge)
      return w;
    var T = g.customMerge(d);
    return typeof T == "function" ? T : w;
  }
  function c(d) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(d).filter(function(g) {
      return Object.propertyIsEnumerable.call(d, g);
    }) : [];
  }
  function p(d) {
    return Object.keys(d).concat(c(d));
  }
  function v(d, g) {
    try {
      return g in d;
    } catch {
      return !1;
    }
  }
  function x(d, g) {
    return v(d, g) && !(Object.hasOwnProperty.call(d, g) && Object.propertyIsEnumerable.call(d, g));
  }
  function h(d, g, T) {
    var y = {};
    return T.isMergeableObject(d) && p(d).forEach(function(_) {
      y[_] = s(d[_], T);
    }), p(g).forEach(function(_) {
      x(d, _) || (v(d, _) && T.isMergeableObject(g[_]) ? y[_] = l(_, T)(d[_], g[_], T) : y[_] = s(g[_], T));
    }), y;
  }
  function w(d, g, T) {
    T = T || {}, T.arrayMerge = T.arrayMerge || u, T.isMergeableObject = T.isMergeableObject || e, T.cloneUnlessOtherwiseSpecified = s;
    var y = Array.isArray(g), _ = Array.isArray(d), P = y === _;
    return P ? y ? T.arrayMerge(d, g, T) : h(d, g, T) : s(g, T);
  }
  w.all = function(g, T) {
    if (!Array.isArray(g))
      throw new Error("first argument should be an array");
    return g.reduce(function(y, _) {
      return w(y, _, T);
    }, {});
  };
  var H = w;
  return hr = H, hr;
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
function $t(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Mr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var G = function() {
  return G = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, G.apply(this, arguments);
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
}, L;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(L || (L = {}));
var Z;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(Z || (Z = {}));
var ht;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(ht || (ht = {}));
function vn(e) {
  return e.type === Z.literal;
}
function ms(e) {
  return e.type === Z.argument;
}
function ti(e) {
  return e.type === Z.number;
}
function ri(e) {
  return e.type === Z.date;
}
function ni(e) {
  return e.type === Z.time;
}
function ii(e) {
  return e.type === Z.select;
}
function ai(e) {
  return e.type === Z.plural;
}
function ps(e) {
  return e.type === Z.pound;
}
function si(e) {
  return e.type === Z.tag;
}
function oi(e) {
  return !!(e && typeof e == "object" && e.type === ht.number);
}
function Pr(e) {
  return !!(e && typeof e == "object" && e.type === ht.dateTime);
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
  for (var t = e.split(bs).filter(function(v) {
    return v.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, c = u; l < c.length; l++) {
      var p = c[l];
      if (p.length === 0)
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
        t = G(G(G({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return G(G({}, u), _n(l));
        }, {}));
        continue;
      case "engineering":
        t = G(G(G({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return G(G({}, u), _n(l));
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
        i.options[0].replace(xs, function(u, l, c, p, v, x) {
          if (l)
            t.minimumIntegerDigits = c.length;
          else {
            if (p && v)
              throw new Error("We currently do not support maximum integer digits");
            if (x)
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
      i.stem.replace(gn, function(u, l, c, p, v, x) {
        return c === "*" ? t.minimumFractionDigits = l.length : p && p[0] === "#" ? t.maximumFractionDigits = p.length : v && x ? (t.minimumFractionDigits = v.length, t.maximumFractionDigits = v.length + x.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = G(G({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = G(G({}, t), bn(a)));
      continue;
    }
    if (ui.test(i.stem)) {
      t = G(G({}, t), bn(i.stem));
      continue;
    }
    var o = ci(i.stem);
    o && (t = G(G({}, t), o));
    var s = Es(i.stem);
    s && (t = G(G({}, t), s));
  }
  return t;
}
var Dt = {
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
  var i = Dt[n || ""] || Dt[r || ""] || Dt["".concat(r, "-001")] || Dt["001"];
  return i[0];
}
var vr, As = new RegExp("^".concat(li.source, "*")), Hs = new RegExp("".concat(li.source, "*$"));
function D(e, t) {
  return { start: e, end: t };
}
var Ms = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ps = !!String.fromCodePoint, Bs = !!Object.fromEntries, Is = !!String.prototype.codePointAt, Ns = !!String.prototype.trimStart, Os = !!String.prototype.trimEnd, Cs = !!Number.isSafeInteger, Ls = Cs ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Br = !0;
try {
  var Rs = di("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Br = ((vr = Rs.exec("a")) === null || vr === void 0 ? void 0 : vr[0]) === "a";
} catch {
  Br = !1;
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
), Ir = Ps ? String.fromCodePoint : (
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
  Bs ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), hi = Is ? (
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
), ks = Ns ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(As, "");
  }
), Ds = Os ? (
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
var Nr;
if (Br) {
  var En = di("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Nr = function(t, r) {
    var n;
    En.lastIndex = r;
    var i = En.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Nr = function(t, r) {
    for (var n = []; ; ) {
      var i = hi(t, r);
      if (i === void 0 || mi(i) || js(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Ir.apply(void 0, n);
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
              type: Z.pound,
              location: D(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(L.UNMATCHED_CLOSING_TAG, D(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Or(this.peek() || 0)) {
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
            type: Z.literal,
            value: "<".concat(i, "/>"),
            location: D(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Or(this.char()))
            return this.error(L.INVALID_TAG, D(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(L.UNMATCHED_CLOSING_TAG, D(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: Z.tag,
              value: i,
              children: o,
              location: D(n, this.clonePosition())
            },
            err: null
          } : this.error(L.INVALID_TAG, D(s, this.clonePosition())));
        } else
          return this.error(L.UNCLOSED_TAG, D(n, this.clonePosition()));
      } else
        return this.error(L.INVALID_TAG, D(n, this.clonePosition()));
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
      var u = D(n, this.clonePosition());
      return {
        val: { type: Z.literal, value: i, location: u },
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
      return Ir.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Ir(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(L.EMPTY_ARGUMENT, D(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(L.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: Z.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: D(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(L.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Nr(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = D(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(L.EXPECT_ARGUMENT_TYPE, D(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var c = this.clonePosition(), p = this.parseSimpleArgStyleIfPossible();
            if (p.err)
              return p;
            var v = Ds(p.val);
            if (v.length === 0)
              return this.error(L.EXPECT_ARGUMENT_STYLE, D(this.clonePosition(), this.clonePosition()));
            var x = D(c, this.clonePosition());
            l = { style: v, styleLocation: x };
          }
          var h = this.tryParseArgumentClose(i);
          if (h.err)
            return h;
          var w = D(i, this.clonePosition());
          if (l && yn(l?.style, "::", 0)) {
            var H = ks(l.style.slice(2));
            if (s === "number") {
              var p = this.parseNumberSkeletonFromString(H, l.styleLocation);
              return p.err ? p : {
                val: { type: Z.number, value: n, location: w, style: p.val },
                err: null
              };
            } else {
              if (H.length === 0)
                return this.error(L.EXPECT_DATE_TIME_SKELETON, w);
              var d = H;
              this.locale && (d = Ts(H, this.locale));
              var v = {
                type: ht.dateTime,
                pattern: d,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? gs(d) : {}
              }, g = s === "date" ? Z.date : Z.time;
              return {
                val: { type: g, value: n, location: w, style: v },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? Z.number : s === "date" ? Z.date : Z.time,
              value: n,
              location: w,
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
            return this.error(L.EXPECT_SELECT_ARGUMENT_OPTIONS, D(T, G({}, T)));
          this.bumpSpace();
          var y = this.parseIdentifierIfPossible(), _ = 0;
          if (s !== "select" && y.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, D(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var p = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, L.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (p.err)
              return p;
            this.bumpSpace(), y = this.parseIdentifierIfPossible(), _ = p.val;
          }
          var P = this.tryParsePluralOrSelectOptions(t, s, r, y);
          if (P.err)
            return P;
          var h = this.tryParseArgumentClose(i);
          if (h.err)
            return h;
          var B = D(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: Z.select,
              value: n,
              options: xn(P.val),
              location: B
            },
            err: null
          } : {
            val: {
              type: Z.plural,
              value: n,
              options: xn(P.val),
              offset: _,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: B
            },
            err: null
          };
        }
        default:
          return this.error(L.INVALID_ARGUMENT_TYPE, D(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, D(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(L.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, D(i, this.clonePosition()));
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
        return this.error(L.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: ht.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? ws(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, c = i.location; ; ) {
        if (l.length === 0) {
          var p = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var v = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_SELECTOR, L.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (v.err)
              return v;
            c = D(p, this.clonePosition()), l = this.message.slice(p.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? L.DUPLICATE_SELECT_ARGUMENT_SELECTOR : L.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, c);
        l === "other" && (o = !0), this.bumpSpace();
        var x = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : L.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, D(this.clonePosition(), this.clonePosition()));
        var h = this.parseMessage(t + 1, r, n);
        if (h.err)
          return h;
        var w = this.tryParseArgumentClose(x);
        if (w.err)
          return w;
        s.push([
          l,
          {
            value: h.val,
            location: D(x, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, c = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR : L.EXPECT_PLURAL_ARGUMENT_SELECTOR, D(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(L.MISSING_OTHER_CLAUSE, D(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      var u = D(i, this.clonePosition());
      return a ? (o *= n, Ls(o) ? { val: o, err: null } : this.error(r, u)) : this.error(t, u);
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
function Or(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Fs(e) {
  return Or(e) || e === 47;
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
function Cr(e) {
  e.forEach(function(t) {
    if (delete t.location, ii(t) || ai(t))
      for (var r in t.options)
        delete t.options[r].location, Cr(t.options[r].value);
    else ti(t) && oi(t.style) || (ri(t) || ni(t)) && Pr(t.style) ? delete t.style.location : si(t) && Cr(t.children);
  });
}
function Vs(e, t) {
  t === void 0 && (t = {}), t = G({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Us(e, t).parse();
  if (r.err) {
    var n = SyntaxError(L[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Cr(r.val), r.val;
}
var dt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(dt || (dt = {}));
var er = (
  /** @class */
  (function(e) {
    $t(t, e);
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
    $t(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), dt.INVALID_VALUE, a) || this;
    }
    return t;
  })(er)
), zs = (
  /** @class */
  (function(e) {
    $t(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), dt.INVALID_VALUE, i) || this;
    }
    return t;
  })(er)
), Xs = (
  /** @class */
  (function(e) {
    $t(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), dt.MISSING_VALUE, n) || this;
    }
    return t;
  })(er)
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
function Gt(e, t, r, n, i, a, o) {
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
    var p = c.value;
    if (!(i && p in i))
      throw new Xs(p, o);
    var v = i[p];
    if (ms(c)) {
      (!v || typeof v == "string" || typeof v == "number") && (v = typeof v == "string" || typeof v == "number" ? String(v) : ""), s.push({
        type: typeof v == "string" ? me.literal : me.object,
        value: v
      });
      continue;
    }
    if (ri(c)) {
      var x = typeof c.style == "string" ? n.date[c.style] : Pr(c.style) ? c.style.parsedOptions : void 0;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, x).format(v)
      });
      continue;
    }
    if (ni(c)) {
      var x = typeof c.style == "string" ? n.time[c.style] : Pr(c.style) ? c.style.parsedOptions : n.time.medium;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, x).format(v)
      });
      continue;
    }
    if (ti(c)) {
      var x = typeof c.style == "string" ? n.number[c.style] : oi(c.style) ? c.style.parsedOptions : void 0;
      x && x.scale && (v = v * (x.scale || 1)), s.push({
        type: me.literal,
        value: r.getNumberFormat(t, x).format(v)
      });
      continue;
    }
    if (si(c)) {
      var h = c.children, w = c.value, H = i[w];
      if (!Ws(H))
        throw new zs(w, "function", o);
      var d = Gt(h, t, r, n, i, a), g = H(d.map(function(_) {
        return _.value;
      }));
      Array.isArray(g) || (g = [g]), s.push.apply(s, g.map(function(_) {
        return {
          type: typeof _ == "string" ? me.literal : me.object,
          value: _
        };
      }));
    }
    if (ii(c)) {
      var T = c.options[v] || c.options.other;
      if (!T)
        throw new wn(c.value, v, Object.keys(c.options), o);
      s.push.apply(s, Gt(T.value, t, r, n, i));
      continue;
    }
    if (ai(c)) {
      var T = c.options["=".concat(v)];
      if (!T) {
        if (!Intl.PluralRules)
          throw new er(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, dt.MISSING_INTL_API, o);
        var y = r.getPluralRules(t, { type: c.pluralType }).select(v - (c.offset || 0));
        T = c.options[y] || c.options.other;
      }
      if (!T)
        throw new wn(c.value, v, Object.keys(c.options), o);
      s.push.apply(s, Gt(T.value, t, r, n, i, v - (c.offset || 0)));
      continue;
    }
  }
  return qs(s);
}
function Zs(e, t) {
  return t ? G(G(G({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = G(G({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ys(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Zs(e[n], t[n]), r;
  }, G({}, e)) : e;
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
        var c = l.reduce(function(p, v) {
          return !p.length || v.type !== me.literal || typeof p[p.length - 1] != "string" ? p.push(v.value) : p[p.length - 1] += v.value, p;
        }, []);
        return c.length <= 1 ? c[0] || "" : c;
      }, this.formatToParts = function(u) {
        return Gt(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
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
        this.ast = e.__parse(t, G(G({}, s), { locale: this.resolvedLocale }));
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
const qe = {}, $s = (e, t, r) => r && (t in qe || (qe[t] = {}), e in qe[t] || (qe[t][e] = r), r), pi = (e, t) => {
  if (t == null)
    return;
  if (t in qe && e in qe[t])
    return qe[t][e];
  const r = tr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = to(i, e);
    if (a)
      return $s(e, t, a);
  }
};
let Wr;
const It = Bt({});
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
  const t = tr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (vi(n))
      return n;
  }
}
function no(e, ...t) {
  delete qe[e], It.update((r) => (r[e] = as.all([r[e] || {}, ...t]), r));
}
pt(
  [It],
  ([e]) => Object.keys(e)
);
It.subscribe((e) => Wr = e);
const jt = {};
function io(e, t) {
  jt[e].delete(t), jt[e].size === 0 && delete jt[e];
}
function gi(e) {
  return jt[e];
}
function ao(e) {
  return tr(e).map((t) => {
    const r = gi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Lr(e) {
  return e == null ? !1 : tr(e).some(
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
  if (!Lr(e))
    return e in Tt ? Tt[e] : Promise.resolve();
  const t = ao(e);
  return Tt[e] = Promise.all(
    t.map(
      ([r, n]) => so(r, n)
    )
  ).then(() => {
    if (Lr(e))
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
function mt() {
  return uo;
}
const br = Bt(!1);
var fo = Object.defineProperty, co = Object.defineProperties, ho = Object.getOwnPropertyDescriptors, Tn = Object.getOwnPropertySymbols, mo = Object.prototype.hasOwnProperty, po = Object.prototype.propertyIsEnumerable, Sn = (e, t, r) => t in e ? fo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, vo = (e, t) => {
  for (var r in t || (t = {}))
    mo.call(t, r) && Sn(e, r, t[r]);
  if (Tn)
    for (var r of Tn(t))
      po.call(t, r) && Sn(e, r, t[r]);
  return e;
}, go = (e, t) => co(e, ho(t));
let Rr;
const Xt = Bt(null);
function An(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function tr(e, t = mt().fallbackLocale) {
  const r = An(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...An(t)])] : r;
}
function tt() {
  return Rr ?? void 0;
}
Xt.subscribe((e) => {
  Rr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const bo = (e) => {
  if (e && ro(e) && Lr(e)) {
    const { loadingDelay: t } = mt();
    let r;
    return typeof window < "u" && tt() != null && t ? r = window.setTimeout(
      () => br.set(!0),
      t
    ) : br.set(!0), bi(e).then(() => {
      Xt.set(e);
    }).finally(() => {
      clearTimeout(r), br.set(!1);
    });
  }
  return Xt.set(e);
}, vt = go(vo({}, Xt), {
  set: bo
}), rr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var _o = Object.defineProperty, qt = Object.getOwnPropertySymbols, _i = Object.prototype.hasOwnProperty, yi = Object.prototype.propertyIsEnumerable, Hn = (e, t, r) => t in e ? _o(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Zr = (e, t) => {
  for (var r in t || (t = {}))
    _i.call(t, r) && Hn(e, r, t[r]);
  if (qt)
    for (var r of qt(t))
      yi.call(t, r) && Hn(e, r, t[r]);
  return e;
}, gt = (e, t) => {
  var r = {};
  for (var n in e)
    _i.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && qt)
    for (var n of qt(e))
      t.indexOf(n) < 0 && yi.call(e, n) && (r[n] = e[n]);
  return r;
};
const Mt = (e, t) => {
  const { formats: r } = mt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, yo = rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = gt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Mt("number", n)), new Intl.NumberFormat(r, i);
  }
), xo = rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = gt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Mt("date", n) : Object.keys(i).length === 0 && (i = Mt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Eo = rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = gt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Mt("time", n) : Object.keys(i).length === 0 && (i = Mt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), wo = (e = {}) => {
  var t = e, {
    locale: r = tt()
  } = t, n = gt(t, [
    "locale"
  ]);
  return yo(Zr({ locale: r }, n));
}, To = (e = {}) => {
  var t = e, {
    locale: r = tt()
  } = t, n = gt(t, [
    "locale"
  ]);
  return xo(Zr({ locale: r }, n));
}, So = (e = {}) => {
  var t = e, {
    locale: r = tt()
  } = t, n = gt(t, [
    "locale"
  ]);
  return Eo(Zr({ locale: r }, n));
}, Ao = rr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = tt()) => new Qs(e, t, mt().formats, {
    ignoreTag: mt().ignoreTag
  })
), Ho = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = tt(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let c = pi(e, u);
  if (!c)
    c = (a = (i = (n = (r = mt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof c != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof c}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), c;
  if (!s)
    return c;
  let p = c;
  try {
    p = Ao(c, u).format(s);
  } catch (v) {
    v instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      v.message
    );
  }
  return p;
}, Mo = (e, t) => So(t).format(e), Po = (e, t) => To(t).format(e), Bo = (e, t) => wo(t).format(e), Io = (e, t = tt()) => pi(e, t);
pt([vt, It], () => Ho);
pt([vt], () => Mo);
pt([vt], () => Po);
pt([vt], () => Bo);
pt([vt, It], () => Io);
const No = "__i18n__", Oo = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Co = [
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
function Lo(e) {
  return typeof e == "string" && e.includes(No);
}
class Ro {
  load_component;
  #t = Y(Ht({}));
  get shared() {
    return f(this.#t);
  }
  set shared(t) {
    A(this.#t, t, !0);
  }
  #r = Y(Ht({}));
  get props() {
    return f(this.#r);
  }
  set props(t) {
    A(this.#r, t, !0);
  }
  #e = Y((t) => t);
  get i18n() {
    return f(this.#e);
  }
  set i18n(t) {
    A(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Co;
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
    for (const n of Oo)
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
    ), Te(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), ne(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && vt.subscribe(() => {
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
      const n = t[r], i = Lo(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    Te(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
da();
var ko = /* @__PURE__ */ qn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Mn = /* @__PURE__ */ fe("<!> <!>", 1), Do = /* @__PURE__ */ fe('<div class="placeholder svelte-1stq1b1"></div>');
function Uo(e, t) {
  Qt(t, !1);
  let r = M(t, "height", 8, void 0), n = M(t, "min_height", 8, void 0), i = M(t, "max_height", 8, void 0), a = M(t, "width", 8, void 0), o = M(t, "elem_id", 8, ""), s = M(t, "elem_classes", 24, () => []), u = M(t, "variant", 8, "solid"), l = M(t, "border_mode", 8, "base"), c = M(t, "padding", 8, !0), p = M(t, "type", 8, "normal"), v = M(t, "test_id", 8, void 0), x = M(t, "explicit_call", 8, !1), h = M(t, "container", 8, !0), w = M(t, "visible", 8, !0), H = M(t, "allow_overflow", 8, !0), d = M(t, "overflow_behavior", 8, "auto"), g = M(t, "scale", 8, null), T = M(t, "min_width", 8, 0), y = M(t, "flex", 12, !1), _ = M(t, "resizable", 8, !1), P = M(t, "rtl", 8, !1), B = M(t, "fullscreen", 12, !1), I = M(t, "label", 8, void 0), k = Je(B()), j = Je(), q = p() === "fieldset" ? "fieldset" : "div", te = Je(0), ie = Je(0), z = Je(null);
  function Ge(re) {
    B() && re.key === "Escape" && B(!1);
  }
  const Se = (re) => {
    if (re !== void 0) {
      if (typeof re == "number")
        return re + "px";
      if (typeof re == "string")
        return re;
    }
  }, je = (re) => {
    let ve = re.clientY;
    const ge = (ue) => {
      const be = ue.clientY - ve;
      ve = ue.clientY, pa(j, f(j).style.height = `${f(j).offsetHeight + be}px`);
    }, xe = () => {
      window.removeEventListener("mousemove", ge), window.removeEventListener("mouseup", xe);
    };
    window.addEventListener("mousemove", ge), window.addEventListener("mouseup", xe);
  };
  nn(
    () => (we(B()), f(k), f(j)),
    () => {
      B() !== f(k) && (A(k, B()), B() ? (A(z, f(j).getBoundingClientRect()), A(te, f(j).offsetHeight), A(ie, f(j).offsetWidth), window.addEventListener("keydown", Ge)) : (A(z, null), window.removeEventListener("keydown", Ge)));
    }
  ), nn(() => we(w()), () => {
    w() || y(!1);
  }), ma(), Ja();
  var rt = lt(), ye = de(rt);
  {
    var bt = (re) => {
      var ve = Mn(), ge = de(ve);
      ka(ge, () => q, !1, (be, Be) => {
        qr(be, (ae) => A(j, ae), () => f(j)), Ya(
          be,
          (ae, Re) => ({
            "data-testid": v(),
            id: o(),
            class: `block ${ae ?? ""}`,
            dir: P() ? "rtl" : "ltr",
            "aria-label": I(),
            style: "",
            [At]: {
              hidden: w() === "hidden",
              padded: c(),
              flex: y(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !x() && !h(),
              fullscreen: B(),
              animating: B() && f(z) !== null,
              "auto-margin": g() === null
            },
            [ot]: Re
          }),
          [
            () => (we(s()), ne(() => s()?.join(" ") || "")),
            () => ({
              height: (we(B()), we(r()), ne(() => B() ? void 0 : Se(r()))),
              "min-height": (we(B()), we(n()), ne(() => B() ? void 0 : Se(n()))),
              "max-height": (we(B()), we(i()), ne(() => B() ? void 0 : Se(i()))),
              "--start-top": (f(z), ne(() => f(z) ? `${f(z).top}px` : "0px")),
              "--start-left": (f(z), ne(() => f(z) ? `${f(z).left}px` : "0px")),
              "--start-width": (f(z), ne(() => f(z) ? `${f(z).width}px` : "0px")),
              "--start-height": (f(z), ne(() => f(z) ? `${f(z).height}px` : "0px")),
              width: (we(B()), we(a()), ne(() => B() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : Se(a()))),
              "border-style": u(),
              overflow: H() ? d() : "hidden",
              "flex-grow": g(),
              "min-width": `calc(min(${T()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Ie = Mn(), Ve = de(Ie);
        Ar(Ve, t, "default", {});
        var Ne = V(Ve, 2);
        {
          var nt = (ae) => {
            var Re = ko();
            Ae("mousedown", Re, je), R(ae, Re);
          };
          J(Ne, (ae) => {
            _() && ae(nt);
          });
        }
        R(Be, Ie);
      });
      var xe = V(ge, 2);
      {
        var ue = (be) => {
          var Be = Do();
          let Ie;
          Q(() => Ie = Pe(Be, "", Ie, {
            height: f(te) + "px",
            width: f(ie) + "px"
          })), R(be, Be);
        };
        J(xe, (be) => {
          B() && be(ue);
        });
      }
      R(re, ve);
    };
    J(ye, (re) => {
      (w() === !0 || w() === "hidden") && re(bt);
    });
  }
  R(e, rt), Jt();
}
var Fo = /* @__PURE__ */ fe('<span class="svelte-vvirtv"> </span>'), Go = /* @__PURE__ */ fe("<button><!> <div><!> <!></div></button>");
function Pn(e, t) {
  let r = M(t, "label", 3, ""), n = M(t, "show_label", 3, !1), i = M(t, "pending", 3, !1), a = M(t, "size", 3, "small"), o = M(t, "padded", 3, !0), s = M(t, "highlight", 3, !1), u = M(t, "disabled", 3, !1), l = M(t, "hasPopup", 3, !1), c = M(t, "color", 3, "var(--block-label-text-color)"), p = M(t, "transparent", 3, !1), v = M(t, "background", 3, "var(--block-background-fill)"), x = M(t, "border", 3, "transparent"), h = He(() => s() ? "var(--color-accent)" : c());
  var w = Go();
  let H, d;
  var g = ee(w);
  {
    var T = (k) => {
      var j = Fo(), q = ee(j);
      Q(() => he(q, r())), R(k, j);
    };
    J(g, (k) => {
      n() && k(T);
    });
  }
  var y = V(g, 2);
  let _;
  var P = ee(y);
  Ca(P, () => t.Icon, (k, j) => {
    j(k, {});
  });
  var B = V(P, 2);
  {
    var I = (k) => {
      var j = lt(), q = de(j);
      Ha(q, () => t.children), R(k, j);
    };
    J(B, (k) => {
      t.children && k(I);
    });
  }
  Q(() => {
    H = Ke(w, 1, "icon-button svelte-vvirtv", null, H, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: p()
    }), w.disabled = u(), ft(w, "aria-label", r()), ft(w, "aria-haspopup", l()), ft(w, "title", r()), d = Pe(w, "", d, {
      "--border-color": x(),
      color: !u() && f(h) ? f(h) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : v()
    }), _ = Ke(y, 1, "svelte-vvirtv", null, _, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), jn("click", w, function(...k) {
    t.onclick?.apply(this, k);
  }), R(e, w);
}
Yt(["click"]);
var jo = /* @__PURE__ */ qn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Bn(e) {
  var t = jo();
  R(e, t);
}
Yt(["click"]);
function _r(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function In(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function kr(e, t, r, n) {
  if (typeof r == "number" || In(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, In(r) ? new Date(r.getTime() + l) : r + l);
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
function Nn(e, t = {}) {
  const r = Bt(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), c = (
    /** @type {T | undefined} */
    e
  ), p = 1, v = 0, x = !1;
  function h(H, d = {}) {
    c = H;
    const g = u = {};
    return e == null || d.hard || w.stiffness >= 1 && w.damping >= 1 ? (x = !0, o = Me.now(), l = H, r.set(e = c), Promise.resolve()) : (d.soft && (v = 1 / ((d.soft === !0 ? 0.5 : +d.soft) * 60), p = 0), s || (o = Me.now(), x = !1, s = Ra((T) => {
      if (x)
        return x = !1, s = null, !1;
      p = Math.min(p + v, 1);
      const y = Math.min(T - o, 1e3 / 30), _ = {
        inv_mass: p,
        opts: w,
        settled: !0,
        dt: y * 60 / 1e3
      }, P = kr(_, l, e, c);
      return o = T, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      P), _.settled && (s = null), !_.settled;
    })), new Promise((T) => {
      s.promise.then(() => {
        g === u && T();
      });
    }));
  }
  const w = {
    set: h,
    update: (H, d) => h(H(
      /** @type {T} */
      c,
      /** @type {T} */
      e
    ), d),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return w;
}
var Vo = /* @__PURE__ */ fe('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function zo(e, t) {
  Qt(t, !0);
  const r = () => an(u, "$top", i), n = () => an(l, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(T, y, _, P) {
    function B(I) {
      return I instanceof _ ? I : new _(function(k) {
        k(I);
      });
    }
    return new (_ || (_ = Promise))(function(I, k) {
      function j(ie) {
        try {
          te(P.next(ie));
        } catch (z) {
          k(z);
        }
      }
      function q(ie) {
        try {
          te(P.throw(ie));
        } catch (z) {
          k(z);
        }
      }
      function te(ie) {
        ie.done ? I(ie.value) : B(ie.value).then(j, q);
      }
      te((P = P.apply(T, y || [])).next());
    });
  };
  let s = M(t, "margin", 3, !0);
  const u = Nn([0, 0]), l = Nn([0, 0]);
  let c = Y(!1);
  function p() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function v() {
    return o(this, void 0, void 0, function* () {
      yield p(), f(c) || v();
    });
  }
  function x() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), v();
    });
  }
  Te(() => (x(), () => {
    A(c, !0);
  }));
  var h = Vo();
  let w;
  var H = ee(h), d = ee(H), g = V(d);
  Q(() => {
    w = Ke(h, 1, "svelte-m6d381", null, w, { margin: s() }), Pe(d, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Pe(g, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), R(e, h), Jt(), a();
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
      } catch (p) {
        o(p);
      }
    }
    function u(c) {
      try {
        l(n.throw(c));
      } catch (p) {
        o(p);
      }
    }
    function l(c) {
      c.done ? a(c.value) : i(c.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let Ut = [], yr = !1;
const qo = typeof window < "u", xi = qo ? window.requestAnimationFrame : (e) => {
};
function Wo(e) {
  return Xo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Ut.push(t), !yr) yr = !0;
      else return;
      yield va(), xi(() => {
        let n = [0, 0];
        for (let i = 0; i < Ut.length; i++) {
          const o = Ut[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), yr = !1, Ut = [];
      });
    }
  });
}
var Zo = /* @__PURE__ */ fe('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), Yo = /* @__PURE__ */ fe('<div class="eta-bar svelte-124hqw6"></div>'), Jo = /* @__PURE__ */ fe("<!> ", 1), Qo = /* @__PURE__ */ fe("<!> <!> <!> <!>", 1), Ko = /* @__PURE__ */ fe('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), $o = /* @__PURE__ */ fe('<p class="loading svelte-124hqw6"> </p> <!>', 1), el = /* @__PURE__ */ fe("<!> <div><!> <!></div> <!> <!>", 1), tl = /* @__PURE__ */ fe('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), rl = /* @__PURE__ */ fe("<div> <!> </div>"), nl = /* @__PURE__ */ fe('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function il(e, t) {
  Qt(t, !0);
  let r = M(t, "eta", 3, null), n = M(t, "scroll_to_output", 3, !1), i = M(t, "timer", 3, !0), a = M(t, "show_progress", 3, "full"), o = M(t, "message", 3, null), s = M(t, "progress", 3, null), u = M(t, "variant", 3, "default"), l = M(t, "loading_text", 3, "Loading..."), c = M(t, "absolute", 3, !0), p = M(t, "translucent", 3, !1), v = M(t, "border", 3, !1), x = M(t, "validation_error", 7, null), h = M(t, "show_validation_error", 3, !0), w = M(t, "type", 3, null), H = M(t, "used_cache", 3, null), d = M(t, "cache_duration", 3, null), g = M(t, "avg_time", 3, null), T, y = !1, _ = Y(0), P = Y(null), B = Y(null), I = Y(!1), k = Y(null), j = Y(!1), q = Y(!1), te = Y(null), ie = Y(null), z = Y("from cache"), Ge = Y(!1), Se = null, je = null;
  const rt = He(() => !(h() && x()) && (w() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let ye = Y(0);
  const bt = He(() => f(B) === null || f(B) <= 0 || !f(ye) ? 0 : Math.min(f(ye) / f(B), 1)), re = He(() => f(ye).toFixed(1));
  let ve = He(() => s() == null), ge = He(() => r() !== null && r() !== void 0 ? r() : f(P));
  function xe() {
    xi(() => {
      A(ye, (performance.now() - f(_)) / 1e3), y && xe();
    });
  }
  let ue = He(() => {
    let X = null;
    s() != null ? X = s().map((oe) => {
      if (oe.index != null && oe.length != null)
        return oe.index / oe.length;
      if (oe.progress != null)
        return oe.progress;
    }) : X = null;
    let K, se = "";
    return X ? (K = X[X.length - 1], K === 0 ? se = "0" : se = "150ms") : K = void 0, {
      progress_level: X,
      last_progress_level: K,
      progress_bar_transition: se
    };
  });
  function be() {
    y || (A(P, A(k, null), !0), A(_, performance.now(), !0), y = !0, xe());
  }
  function Be() {
    A(P, A(k, null), !0), y && (y = !1);
  }
  Te(() => {
    t.status === "pending" ? be() : ne(() => {
      Be();
    });
  }), Te(() => {
    T && n() && (t.status === "pending" || t.status === "complete") && Wo(T, t.autoscroll);
  }), Te(() => {
    f(ge) != null && f(P) !== f(ge) && (A(B, (performance.now() - f(_)) / 1e3 + f(ge)), A(k, f(B).toFixed(1), !0), A(P, f(ge), !0));
  });
  function Ie() {
    A(I, !1);
  }
  Te(() => {
    ne(() => {
      Ie();
    }), t.status === "error" && o() && A(I, !0);
  }), Te(() => {
    t.status === "complete" && w() === "output" && H() && d() != null && (A(te, d().toFixed(1), !0), A(z, H() === "full" ? "from cache" : "used cache", !0), A(Ge, g() != null && g() > d() && g() > 0, !0), A(ie, f(Ge) ? g().toFixed(1) : null, !0), A(j, !0), A(q, !1), Se && clearTimeout(Se), je && clearTimeout(je), Se = setTimeout(
      () => {
        A(q, !0), je = setTimeout(
          () => {
            A(j, !1), A(q, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var Ve = nl(), Ne = de(Ve);
  let nt, ae;
  var Re = ee(Ne);
  {
    var ze = (X) => {
      var K = Zo(), se = ee(K), oe = V(se), _e = ee(oe);
      {
        let m = He(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Pn(_e, {
          get Icon() {
            return Bn;
          },
          get label() {
            return f(m);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => x(null)
        });
      }
      Q(() => he(se, `${x() ?? ""} `)), R(X, K);
    };
    J(Re, (X) => {
      x() && h() && X(ze);
    });
  }
  var nr = V(Re, 2);
  {
    var Nt = (X) => {
      var K = el(), se = de(K);
      {
        var oe = (O) => {
          var F = Yo();
          let le;
          Q(() => le = Pe(F, "", le, {
            transform: `translateX(${(f(bt) || 0) * 100 - 100}%)`
          })), R(O, F);
        };
        J(se, (O) => {
          u() === "default" && f(ve) && a() === "full" && O(oe);
        });
      }
      var _e = V(se, 2);
      let m;
      var b = ee(_e);
      {
        var S = (O) => {
          var F = lt(), le = de(F);
          ln(le, 17, s, sn, (ke, Ce) => {
            var Ct = lt(), sr = de(Ct);
            {
              var Lt = (Ze) => {
                var _t = Jo(), Rt = de(_t);
                {
                  var or = (De) => {
                    var Ye = Le();
                    Q((yt, xt) => he(Ye, `${yt ?? ""}/${xt ?? ""}`), [
                      () => _r(f(Ce).index || 0),
                      () => _r(f(Ce).length)
                    ]), R(De, Ye);
                  }, it = (De) => {
                    var Ye = Le();
                    Q((yt) => he(Ye, yt), [() => _r(f(Ce).index || 0)]), R(De, Ye);
                  };
                  J(Rt, (De) => {
                    f(Ce).length != null ? De(or) : De(it, -1);
                  });
                }
                var at = V(Rt);
                Q(() => he(at, ` ${f(Ce).unit ?? ""} |  `)), R(Ze, _t);
              };
              J(sr, (Ze) => {
                f(Ce).index != null && Ze(Lt);
              });
            }
            R(ke, Ct);
          }), R(O, F);
        }, E = (O) => {
          var F = Le();
          Q(() => he(F, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), R(O, F);
        }, U = (O) => {
          var F = Le("processing |");
          R(O, F);
        };
        J(b, (O) => {
          s() ? O(S) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? O(E, 1) : t.queue_position === 0 && O(U, 2);
        });
      }
      var N = V(b, 2);
      {
        var C = (O) => {
          var F = Le();
          Q(() => he(F, `${f(re) ?? ""}${r() ? `/${f(k)}` : ""}s`)), R(O, F);
        };
        J(N, (O) => {
          i() && O(C);
        });
      }
      var W = V(_e, 2);
      {
        var $ = (O) => {
          var F = Ko(), le = ee(F), ke = ee(le);
          {
            var Ce = (Ze) => {
              var _t = lt(), Rt = de(_t);
              ln(Rt, 17, s, sn, (or, it, at) => {
                var De = lt(), Ye = de(De);
                {
                  var yt = (xt) => {
                    var Yr = Qo(), Jr = de(Yr);
                    {
                      var Ei = (ce) => {
                        var Ue = Le(" /");
                        R(ce, Ue);
                      };
                      J(Jr, (ce) => {
                        at !== 0 && ce(Ei);
                      });
                    }
                    var Qr = V(Jr, 2);
                    {
                      var wi = (ce) => {
                        var Ue = Le();
                        Q(() => he(Ue, f(it).desc)), R(ce, Ue);
                      };
                      J(Qr, (ce) => {
                        f(it).desc != null && ce(wi);
                      });
                    }
                    var Kr = V(Qr, 2);
                    {
                      var Ti = (ce) => {
                        var Ue = Le("-");
                        R(ce, Ue);
                      };
                      J(Kr, (ce) => {
                        f(it).desc != null && f(ue).progress_level && f(ue).progress_level[at] != null && ce(Ti);
                      });
                    }
                    var Si = V(Kr, 2);
                    {
                      var Ai = (ce) => {
                        var Ue = Le();
                        Q((Hi) => he(Ue, `${Hi ?? ""}%`), [
                          () => (100 * (f(ue).progress_level[at] || 0)).toFixed(1)
                        ]), R(ce, Ue);
                      };
                      J(Si, (ce) => {
                        f(ue).progress_level != null && ce(Ai);
                      });
                    }
                    R(xt, Yr);
                  };
                  J(Ye, (xt) => {
                    (f(it).desc != null || f(ue).progress_level && f(ue).progress_level[at] != null) && xt(yt);
                  });
                }
                R(or, De);
              }), R(Ze, _t);
            };
            J(ke, (Ze) => {
              s() != null && Ze(Ce);
            });
          }
          var Ct = V(le, 2), sr = ee(Ct);
          let Lt;
          Q(() => Lt = Pe(sr, "", Lt, {
            width: `${f(ue).last_progress_level * 100}%`,
            transition: f(ue).progress_bar_transition
          })), R(O, F);
        }, Oe = (O) => {
          {
            let F = He(() => u() === "default");
            zo(O, {
              get margin() {
                return f(F);
              }
            });
          }
        };
        J(W, (O) => {
          f(ue).last_progress_level != null ? O($) : a() === "full" && O(Oe, 1);
        });
      }
      var pe = V(W, 2);
      {
        var Ee = (O) => {
          var F = $o(), le = de(F), ke = ee(le), Ce = V(le, 2);
          Ar(Ce, t, "additional-loading-text", {}), Q(() => he(ke, l())), R(O, F);
        };
        J(pe, (O) => {
          i() || O(Ee);
        });
      }
      Q(() => m = Ke(_e, 1, "progress-text svelte-124hqw6", null, m, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), R(X, K);
    }, ir = (X) => {
      var K = tl(), se = de(K), oe = ee(se);
      {
        let S = He(() => t.i18n("common.clear"));
        Pn(oe, {
          get Icon() {
            return Bn;
          },
          get label() {
            return f(S);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var _e = V(se, 2), m = ee(_e), b = V(_e, 2);
      Ar(b, t, "error", {}), Q((S) => he(m, S), [() => t.i18n("common.error")]), R(X, K);
    };
    J(nr, (X) => {
      t.status === "pending" ? X(Nt) : t.status === "error" && X(ir, 1);
    });
  }
  qr(Ne, (X) => T = X, () => T);
  var ar = V(Ne, 2);
  {
    var Ot = (X) => {
      var K = rl();
      let se, oe;
      var _e = ee(K), m = V(_e);
      {
        var b = (E) => {
          var U = Le();
          Q(() => he(U, `~${f(ie) ?? ""}s
			→ `)), R(E, U);
        };
        J(m, (E) => {
          f(Ge) && E(b);
        });
      }
      var S = V(m);
      Q(() => {
        se = Ke(K, 1, "cache-indicator svelte-124hqw6", null, se, { "fade-out": f(q) }), oe = Pe(K, "", oe, { position: c() ? "absolute" : "static" }), he(_e, `⚡ ${f(z) ?? ""}: `), he(S, `${f(te) ?? ""}s`);
      }), R(X, K);
    };
    J(ar, (X) => {
      f(j) && X(Ot);
    });
  }
  Q(() => {
    nt = Ke(Ne, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", nt, {
      "no-click": x() && h(),
      hide: f(rt),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || p() || a() === "minimal" || x(),
      generating: t.status === "generating" && a() === "full",
      border: v()
    }), ae = Pe(Ne, "", ae, {
      position: c() ? "absolute" : "static",
      padding: c() ? "0" : "var(--size-8) 0"
    });
  }), R(e, Ve), Jt();
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
Yt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var ul = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), fl = /* @__PURE__ */ fe('<!> <div class="layout-editor svelte-r41nsf"><div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">重置</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回视野</button></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function hl(e, t) {
  Qt(t, !0);
  const r = /* @__PURE__ */ Ka(t, ul), n = 2048, i = new Ro(r);
  let a, o = null, s = null, u = null, l = null, c = Y(!1), p = Y(!1), v = Y("等待版图 mask"), x = Y(Ht({ enabled: !1 })), h = Y(Ht(P())), w = Y(!1), H = Y(!1), d = Y("crosshair"), g = "", T = { x: 0, y: 0, center_x: 0, center_y: 0 }, y = { angle: 0, rotation: 0 }, _ = null;
  function P() {
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
  function B(m) {
    return JSON.parse(JSON.stringify(m || { enabled: !1 }));
  }
  function I(m) {
    return typeof m == "number" ? `${m}px` : m || "520px";
  }
  function k(m, b, S) {
    return Math.max(b, Math.min(S, m));
  }
  function j(m) {
    let b = ((m + 180) % 360 + 360) % 360 - 180;
    return b === -180 && (b = 180), b;
  }
  function q() {
    return Math.max(1, Number(f(x).target_width || o?.naturalWidth || 1));
  }
  function te() {
    return Math.max(1, Number(f(x).target_height || o?.naturalHeight || 1));
  }
  function ie() {
    return Math.max(1, Number(f(x).source_width || s?.naturalWidth || 1));
  }
  function z() {
    return Math.max(1, Number(f(x).source_height || s?.naturalHeight || 1));
  }
  function Ge() {
    return Math.min(1, n / Math.max(q(), te()));
  }
  function Se() {
    const m = f(x).foreground_bbox_xyxy;
    return Array.isArray(m) && m.length >= 4 ? m.map(Number) : [0, 0, ie() - 1, z() - 1];
  }
  function je(m, b) {
    if (!m) {
      b(null);
      return;
    }
    const S = new Image();
    S.onload = () => b(S), S.onerror = () => b(null), S.src = m;
  }
  function rt() {
    if (!s) {
      u = null, l = null;
      return;
    }
    const m = ie(), b = z();
    u = document.createElement("canvas"), u.width = m, u.height = b;
    const S = u.getContext("2d", { willReadFrequently: !0 });
    if (!S) return;
    S.imageSmoothingEnabled = !1, S.drawImage(s, 0, 0, m, b);
    const E = S.getImageData(0, 0, m, b);
    l = document.createElement("canvas"), l.width = m, l.height = b;
    const U = l.getContext("2d");
    if (!U) return;
    const N = U.createImageData(m, b);
    for (let C = 0; C < E.data.length; C += 4) {
      const W = Math.max(E.data[C], E.data[C + 1], E.data[C + 2]);
      E.data[C + 3] > 0 && W >= 128 && (N.data[C] = 0, N.data[C + 1] = 255, N.data[C + 2] = 120, N.data[C + 3] = 255);
    }
    U.putImageData(N, 0, 0);
  }
  function ye() {
    _ && (clearTimeout(_), _ = null);
  }
  function bt(m) {
    ye(), A(x, B(m), !0), A(h, Object.assign(Object.assign({}, P()), f(x).transform || {}), !0), A(v, f(x).status || "编辑器已加载", !0), A(c, !1), A(p, !1), je(f(x).base_image, (b) => {
      o = b, A(c, !!b), ze();
    }), je(f(x).mask_image, (b) => {
      s = b, A(p, !!b), rt(), ze();
    });
  }
  Te(() => {
    const m = JSON.stringify(i.props.value || null);
    m !== g && (g = m, bt(i.props.value));
  }), Pa(() => {
    ye();
  });
  function re(m) {
    const b = Number(m.rotation_deg || 0) * Math.PI / 180, S = Number(m.scale || 1), E = Math.cos(b), U = Math.sin(b), N = S * E, C = S * U, W = Number(m.center_x || 0) - N * Number(m.pivot_x || 0) + C * Number(m.pivot_y || 0), $ = Number(m.center_y || 0) - C * Number(m.pivot_x || 0) - N * Number(m.pivot_y || 0);
    return [N, C, -C, N, W, $];
  }
  function ve(m, b, S = f(h)) {
    const [E, U, N, C, W, $] = re(S);
    return { x: E * m + N * b + W, y: U * m + C * b + $ };
  }
  function ge(m, b, S = f(h)) {
    const [E, U, N, C, W, $] = re(S), Oe = E * C - U * N;
    if (Math.abs(Oe) < 1e-9) return { x: -1, y: -1 };
    const pe = m - W, Ee = b - $;
    return { x: (C * pe - N * Ee) / Oe, y: (-U * pe + E * Ee) / Oe };
  }
  function xe(m) {
    const b = a.getBoundingClientRect();
    return {
      x: (m.clientX - b.left) / Math.max(1, b.width) * q(),
      y: (m.clientY - b.top) / Math.max(1, b.height) * te()
    };
  }
  function ue(m, b) {
    if (!u) return !1;
    const S = Math.round(m), E = Math.round(b);
    if (S < 0 || E < 0 || S >= u.width || E >= u.height) return !1;
    const U = u.getContext("2d", { willReadFrequently: !0 });
    if (!U) return !1;
    const N = U.getImageData(S, E, 1, 1).data;
    return N[3] > 0 && Math.max(N[0], N[1], N[2]) >= 128;
  }
  function be(m, b) {
    const S = ge(m, b);
    return ue(S.x, S.y);
  }
  function Be() {
    const [m, b, S, E] = Se();
    return [
      ve(m, b),
      ve(S, b),
      ve(S, E),
      ve(m, E)
    ];
  }
  function Ie() {
    const m = Be();
    let b = m[0];
    for (const N of m)
      (N.y < b.y || Math.abs(N.y - b.y) < 1e-6 && N.x > b.x) && (b = N);
    const S = Number(f(h).rotation_deg || 0) * Math.PI / 180, E = Math.cos(S - Math.PI / 4), U = Math.sin(S - Math.PI / 4);
    return { x: b.x + E * 34, y: b.y + U * 34 };
  }
  function Ve() {
    const m = a?.getBoundingClientRect();
    return m ? Math.max(8, 14 * q() / Math.max(1, m.width)) : 14;
  }
  function Ne(m, b) {
    const S = Ie(), E = Ve();
    return Math.hypot(m - S.x, b - S.y) <= E;
  }
  function nt(m) {
    A(
      h,
      Object.assign(Object.assign({}, f(h)), {
        revision: Number(f(h).revision || 0) + 1,
        origin: m,
        scale: k(Number(f(h).scale || 1), 0.01, 20),
        rotation_deg: j(Number(f(h).rotation_deg || 0))
      }),
      !0
    );
  }
  function ae(m, b, S = !0) {
    nt(m), A(
      x,
      Object.assign(Object.assign({}, f(x)), {
        enabled: !0,
        transform: Object.assign({}, f(h)),
        status: b
      }),
      !0
    ), A(v, b, !0), i.props.value = f(x), g = JSON.stringify(f(x)), S && (ye(), i.dispatch("change")), ze();
  }
  function Re(m, b, S = 140) {
    ae(m, b, !1), ye(), _ = setTimeout(
      () => {
        _ = null, i.dispatch("change");
      },
      S
    );
  }
  function ze() {
    if (!a) return;
    const m = q(), b = te(), S = Ge();
    a.width = Math.max(1, Math.round(m * S)), a.height = Math.max(1, Math.round(b * S));
    const E = a.getContext("2d");
    if (E && (E.setTransform(S, 0, 0, S, 0, 0), E.clearRect(0, 0, m, b), o && f(c) ? (E.imageSmoothingEnabled = !0, E.drawImage(o, 0, 0, m, b)) : (E.fillStyle = "#f8fafc", E.fillRect(0, 0, m, b), E.fillStyle = "#64748b", E.font = "18px sans-serif", E.fillText("请先上传图像", 24, 42)), l && f(p) && f(x).enabled !== !1)) {
      E.save(), E.globalAlpha = k(Number(f(h).preview_alpha || 0.35), 0, 1);
      const [U, N, C, W, $, Oe] = re(f(h));
      E.setTransform(S * U, S * N, S * C, S * W, S * $, S * Oe), E.imageSmoothingEnabled = !1, E.drawImage(l, 0, 0, ie(), z()), E.restore();
      const pe = Be();
      E.save(), E.lineJoin = "round", E.lineWidth = Math.max(2.5, m / 700), E.strokeStyle = "rgba(0,0,0,0.82)", E.beginPath(), E.moveTo(pe[0].x, pe[0].y);
      for (let F = 1; F < pe.length; F++) E.lineTo(pe[F].x, pe[F].y);
      E.closePath(), E.stroke(), E.lineWidth = Math.max(1.8, m / 1e3), E.strokeStyle = "#00ff66", E.stroke();
      const Ee = Ie(), O = pe.reduce((F, le) => le.y < F.y || Math.abs(le.y - F.y) < 1e-6 && le.x > F.x ? le : F, pe[0]);
      E.strokeStyle = "#0f172a", E.lineWidth = Math.max(2, m / 900), E.beginPath(), E.moveTo(O.x, O.y), E.lineTo(Ee.x, Ee.y), E.stroke(), E.fillStyle = f(H) ? "#ffb000" : "#ffffff", E.strokeStyle = "#00ff66", E.lineWidth = Math.max(2, m / 900), E.beginPath(), E.arc(Ee.x, Ee.y, Ve(), 0, Math.PI * 2), E.fill(), E.stroke(), E.restore();
    }
  }
  function nr(m) {
    if (m.button !== 0) return;
    if (!f(x).enabled || !f(p) || !u) {
      A(v, "请先启用并加载版图 mask"), ze();
      return;
    }
    ye();
    const b = xe(m);
    if (Ne(b.x, b.y)) {
      A(d, "grabbing"), A(H, !0), y = {
        angle: Math.atan2(b.y - f(h).center_y, b.x - f(h).center_x) * 180 / Math.PI,
        rotation: Number(f(h).rotation_deg || 0)
      }, a.setPointerCapture(m.pointerId);
      return;
    }
    if (!be(b.x, b.y)) {
      A(v, "请点中版图 mask 前景后拖动"), ze();
      return;
    }
    A(d, "grabbing"), A(w, !0), T = {
      x: b.x,
      y: b.y,
      center_x: Number(f(h).center_x || 0),
      center_y: Number(f(h).center_y || 0)
    }, a.setPointerCapture(m.pointerId);
  }
  function Nt(m) {
    if (!f(x).enabled || !f(p) || !u) {
      A(d, "not-allowed");
      return;
    }
    if (Ne(m.x, m.y) || be(m.x, m.y)) {
      A(d, "grab");
      return;
    }
    A(d, "crosshair");
  }
  function ir(m) {
    const b = xe(m);
    if (!f(w) && !f(H)) {
      Nt(b);
      return;
    }
    if (A(d, "grabbing"), f(w))
      A(
        h,
        Object.assign(Object.assign({}, f(h)), {
          center_x: T.center_x + b.x - T.x,
          center_y: T.center_y + b.y - T.y
        }),
        !0
      ), A(v, "正在拖动；松开后同步变换");
    else if (f(H)) {
      const S = Math.atan2(b.y - f(h).center_y, b.x - f(h).center_x) * 180 / Math.PI;
      A(
        h,
        Object.assign(Object.assign({}, f(h)), {
          rotation_deg: j(y.rotation + S - y.angle)
        }),
        !0
      ), A(v, "正在旋转；松开后同步变换");
    }
    ze();
  }
  function ar() {
    !f(w) && !f(H) && A(d, "crosshair");
  }
  function Ot(m) {
    if (f(w) || f(H)) {
      A(w, !1), A(H, !1);
      try {
        a.releasePointerCapture(m.pointerId);
      } catch {
      }
      Nt(xe(m)), ae("canvas", "Canvas 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function X(m) {
    if (!f(x).enabled || !f(p)) return;
    m.preventDefault();
    const b = xe(m), S = ge(b.x, b.y), E = Math.exp(-m.deltaY * 1e-3), U = k(Number(f(h).scale || 1) * E, 0.01, 20);
    let N = Object.assign(Object.assign({}, f(h)), { scale: U });
    const C = ve(S.x, S.y, N);
    N = Object.assign(Object.assign({}, N), {
      center_x: Number(N.center_x || 0) + b.x - C.x,
      center_y: Number(N.center_y || 0) + b.y - C.y
    }), A(h, N, !0), Re("canvas", `滚轮缩放已同步: scale=${U.toFixed(3)}`);
  }
  function K() {
    A(
      h,
      Object.assign(Object.assign({}, f(h)), {
        center_x: q() / 2,
        center_y: te() / 2,
        scale: 1,
        rotation_deg: 0
      }),
      !0
    ), ae("reset", "重置：已居中，scale=1，rotation=0");
  }
  function se() {
    A(h, Object.assign(Object.assign({}, f(h)), { center_x: q() / 2, center_y: te() / 2 }), !0), ae("center", "居中：保留缩放和旋转");
  }
  function oe() {
    const [m, b, S, E] = Se(), U = Math.max(1, S - m + 1), N = Math.max(1, E - b + 1), C = k(Math.min(q() / U, te() / N) * 0.9, 0.01, 20);
    A(
      h,
      Object.assign(Object.assign({}, f(h)), {
        center_x: q() / 2,
        center_y: te() / 2,
        scale: C
      }),
      !0
    ), ae("fit", `适配：scale=${C.toFixed(3)}`);
  }
  function _e() {
    const m = Be(), b = Math.min(...m.map(($) => $.x)), S = Math.max(...m.map(($) => $.x)), E = Math.min(...m.map(($) => $.y)), U = Math.max(...m.map(($) => $.y));
    let N = 0, C = 0;
    const W = Math.max(20, q() * 0.03);
    if (S < W ? N = W - S : b > q() - W && (N = q() - W - b), U < W ? C = W - U : E > te() - W && (C = te() - W - E), N === 0 && C === 0) {
      A(v, "版图已经在视野内");
      return;
    }
    A(
      h,
      Object.assign(Object.assign({}, f(h)), {
        center_x: Number(f(h).center_x || 0) + N,
        center_y: Number(f(h).center_y || 0) + C
      }),
      !0
    ), ae("bring_into_view", "已找回到视野内");
  }
  {
    let m = He(() => f(w) || f(H) ? "focus" : "base");
    Uo(e, {
      get visible() {
        return i.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return f(m);
      },
      padding: !1,
      get elem_id() {
        return i.shared.elem_id;
      },
      get elem_classes() {
        return i.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return i.shared.container;
      },
      get scale() {
        return i.shared.scale;
      },
      get min_width() {
        return i.shared.min_width;
      },
      children: (b, S) => {
        var E = fl(), U = de(E);
        il(U, es(
          {
            get autoscroll() {
              return i.shared.autoscroll;
            },
            get i18n() {
              return i.i18n;
            }
          },
          () => i.shared.loading_status,
          {
            on_clear_status: () => i.dispatch("clear_status", i.shared.loading_status)
          }
        ));
        var N = V(U, 2), C = ee(N), W = ee(C), $ = V(W, 2), Oe = V($, 2), pe = V(Oe, 2), Ee = V(C, 2), O = ee(Ee);
        qr(O, (ke) => a = ke, () => a);
        var F = V(Ee, 2), le = ee(F);
        Q(
          (ke) => {
            Pe(N, ke), Pe(O, `cursor:${f(d)}`), he(le, f(v));
          },
          [() => `min-height:${I(i.props.height)}`]
        ), Ae("click", W, K), Ae("click", $, se), Ae("click", Oe, oe), Ae("click", pe, _e), Ae("pointerdown", O, nr), Ae("pointermove", O, ir), Ae("pointerup", O, Ot), Ae("pointercancel", O, Ot), Ae("pointerleave", O, ar), Ae("wheel", O, X), R(b, E);
      },
      $$slots: { default: !0 }
    });
  }
  Jt();
}
export {
  hl as default
};
