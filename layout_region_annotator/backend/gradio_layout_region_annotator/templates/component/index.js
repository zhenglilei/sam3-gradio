import { i as Cr, g as Bn, o as Ai, n as je, u as te, s as Hi, r as br, m as We, a as w, b as h, t as Rr, d as Pi, q as Ii, c as On, e as Ye, f as zt, h as Ft, j as Bi, T as Oi, k as Li, l as Gt, p as Ze, v as Dr, w as Je, x as Ln, y as Nn, z as Mn, A as At, E as Xt, B as Jr, C as Ni, D as Cn, F as kr, G as Mi, H as Qr, I as Ci, J as Ri, K as Re, L as Rn, M as ar, N as Di, O as ki, P as Ui, Q as Fi, R as Dn, S as Ur, U as Kr, V as $r, W as Gi, X as ji, Y as Vi, Z as zi, _ as Xi, $ as qi, a0 as Wi, a1 as Zi, a2 as Fr, a3 as Yi, a4 as kn, a5 as qt, a6 as Ji, a7 as Qi, a8 as Ki, a9 as $i, aa as Un, ab as ea, ac as ta, ad as Gr, ae as ra, af as _e, ag as na, ah as me, ai as _r, aj as yr, ak as ia, al as aa, am as Tt, an as sa, ao as oa, ap as la, aq as ua, ar as fa, as as ha, at as Fn, au as _t, av as ca, aw as en, ax as da, ay as se, az as Wt, aA as Zt, aB as j, aC as Ce, aD as Y, aE as pa, aF as K, aG as ae, aH as be, aI as q, aJ as va, aK as ma } from "./render-DNiZdw6o.js";
const ga = [];
function ba(e, t = !1, r = !1) {
  return Dt(e, /* @__PURE__ */ new Map(), "", ga, null, r);
}
function Dt(e, t, r, n, i = null, a = !1) {
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
    if (Cr(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var u = 0; u < e.length; u += 1) {
        var l = e[u];
        u in e && (s[u] = Dt(l, t, r, n, null, a));
      }
      return s;
    }
    if (Bn(e) === Ai) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var f of Object.keys(e))
        s[f] = Dt(
          // @ts-expect-error
          e[f],
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
      return Dt(
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
function jr(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), je;
  const n = te(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const rt = [];
function _a(e, t) {
  return {
    subscribe: Ht(e, t).subscribe
  };
}
function Ht(e, t = je) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Hi(e, s) && (e = s, r)) {
      const u = !rt.length;
      for (const l of n)
        l[1](), rt.push(l, e);
      if (u) {
        for (let l = 0; l < rt.length; l += 2)
          rt[l][0](rt[l + 1]);
        rt.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = je) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || je), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function ft(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return _a(r, (o, s) => {
    let u = !1;
    const l = [];
    let f = 0, p = je;
    const v = () => {
      if (f)
        return;
      p();
      const m = t(n ? l[0] : l, o, s);
      a ? o(m) : p = typeof m == "function" ? m : je;
    }, E = i.map(
      (m, y) => jr(
        m,
        (S) => {
          l[y] = S, f &= ~(1 << y), u && v();
        },
        () => {
          f |= 1 << y;
        }
      )
    );
    return u = !0, v(), function() {
      br(E), p(), u = !1;
    };
  });
}
function ya(e) {
  let t;
  return jr(e, (r) => t = r)(), t;
}
let Mt = !1, xr = /* @__PURE__ */ Symbol("unmounted");
function tn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: We(void 0),
    unsubscribe: je
  };
  if (n.store !== e && !(xr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = je;
    else {
      var i = !0;
      n.unsubscribe = jr(e, (a) => {
        i ? n.source.v = a : w(n.source, a);
      }), i = !1;
    }
  return e && xr in r ? ya(e) : h(n.source);
}
function xa() {
  const e = {};
  function t() {
    Rr(() => {
      for (var r in e)
        e[r].unsubscribe();
      Pi(e, xr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = Mt;
  try {
    return Mt = !1, [e(), Mt];
  } finally {
    Mt = t;
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
function Gn(e) {
  var t = On("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function st(e, t) {
  var r = (
    /** @type {Effect} */
    zt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function re(e, t) {
  var r = (t & Oi) !== 0, n = (t & Li) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Gn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    Ft(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Bi ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        Ft(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      st(s, u);
    } else
      st(o, o);
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
        Gn(i)
      ), s = (
        /** @type {Element} */
        Ft(o)
      );
      a = /** @type {Element} */
      Ft(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return st(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function jn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function Pe(e = "") {
  {
    var t = Ye(e + "");
    return st(t, t), t;
  }
}
function it() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = Ye();
  return e.append(t, r), st(t, r), e;
}
function C(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Yt {
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
        Gt(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Gt(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (Ze(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Nn(o, l), l.append(Ye()), this.#e.set(a, { effect: o, fragment: l });
          } else
            Ze(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Dr(o, s, !1)) : s();
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
      r.includes(n) || (Ze(i.effect), this.#e.delete(n));
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
    ), i = Mn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = Ye();
        a.append(o), this.#e.set(t, {
          effect: Je(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          Je(() => r(this.anchor))
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
  var n = new Yt(e);
  At(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Xt);
}
function W(e, t, r = !1) {
  var n = new Yt(e), i = r ? Xt : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  At(() => {
    var o = !1;
    t((s, u = 0) => {
      o = !0, a(u, s);
    }), o || a(-1, null);
  }, i);
}
function rn(e, t) {
  return t;
}
function Pa(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let p = t[s];
    Dr(
      p,
      () => {
        if (a) {
          if (a.pending.delete(p), a.done.add(p), a.pending.size === 0) {
            var v = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Er(e, kr(a.done)), v.delete(a), v.size === 0 && (e.outrogroups = null);
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
      ), f = (
        /** @type {Element} */
        l.parentNode
      );
      ki(f), f.append(l), e.items.clear();
    }
    Er(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Er(e, t, r = !0) {
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
      a.f |= Re;
      const o = document.createDocumentFragment();
      Nn(a, o);
    } else
      Ze(t[i], r);
  }
}
var nn;
function an(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = Cn(() => {
    var d = r();
    return (
      /** @type {V[]} */
      Cr(d) ? d : d == null ? [] : kr(d)
    );
  }), f, p = /* @__PURE__ */ new Map(), v = !0;
  function E(d) {
    (S.effect.f & Rn) === 0 && (S.pending.delete(d), S.fallback = u, Ia(S, f, o, t, n), u !== null && (f.length === 0 ? (u.f & Re) === 0 ? Gt(u) : (u.f ^= Re, Et(u, null, o)) : Dr(u, () => {
      u = null;
    })));
  }
  function m(d) {
    S.pending.delete(d);
  }
  var y = At(() => {
    f = /** @type {V[]} */
    h(l);
    for (var d = f.length, c = /* @__PURE__ */ new Set(), b = (
      /** @type {Batch} */
      Ln
    ), g = Mn(), _ = 0; _ < d; _ += 1) {
      var P = f[_], I = n(P, _), B = v ? null : s.get(I);
      B ? (B.v && Jr(B.v, P), B.i && Jr(B.i, _), g && b.unskip_effect(B.e)) : (B = Ba(
        s,
        v ? o : nn ??= Ye(),
        P,
        I,
        _,
        i,
        t,
        r
      ), v || (B.e.f |= Re), s.set(I, B)), c.add(I);
    }
    if (d === 0 && a && !u && (v ? u = Je(() => a(o)) : (u = Je(() => a(nn ??= Ye())), u.f |= Re)), d > c.size && Ni(), !v)
      if (p.set(b, c), g) {
        for (const [N, R] of s)
          c.has(N) || b.skip_effect(R.e);
        b.oncommit(E), b.ondiscard(m);
      } else
        E(b);
    h(l);
  }), S = { effect: y, items: s, pending: p, outrogroups: null, fallback: u };
  v = !1;
}
function yt(e) {
  for (; e !== null && (e.f & Di) === 0; )
    e = e.next;
  return e;
}
function Ia(e, t, r, n, i) {
  var a = t.length, o = e.items, s = yt(e.effect.first), u, l = null, f = [], p = [], v, E, m, y;
  for (y = 0; y < a; y += 1) {
    if (v = t[y], E = i(v, y), m = /** @type {EachItem} */
    o.get(E).e, e.outrogroups !== null)
      for (const B of e.outrogroups)
        B.pending.delete(m), B.done.delete(m);
    if ((m.f & ar) !== 0 && Gt(m), (m.f & Re) !== 0)
      if (m.f ^= Re, m === s)
        Et(m, null, r);
      else {
        var S = l ? l.next : s;
        m === e.effect.last && (e.effect.last = m.prev), m.prev && (m.prev.next = m.next), m.next && (m.next.prev = m.prev), Fe(e, l, m), Fe(e, m, S), Et(m, S, r), l = m, f = [], p = [], s = yt(l.next);
        continue;
      }
    if (m !== s) {
      if (u !== void 0 && u.has(m)) {
        if (f.length < p.length) {
          var d = p[0], c;
          l = d.prev;
          var b = f[0], g = f[f.length - 1];
          for (c = 0; c < f.length; c += 1)
            Et(f[c], d, r);
          for (c = 0; c < p.length; c += 1)
            u.delete(p[c]);
          Fe(e, b.prev, g.next), Fe(e, l, b), Fe(e, g, d), s = d, l = g, y -= 1, f = [], p = [];
        } else
          u.delete(m), Et(m, s, r), Fe(e, m.prev, m.next), Fe(e, m, l === null ? e.effect.first : l.next), Fe(e, l, m), l = m;
        continue;
      }
      for (f = [], p = []; s !== null && s !== m; )
        (u ??= /* @__PURE__ */ new Set()).add(s), p.push(s), s = yt(s.next);
      if (s === null)
        continue;
    }
    (m.f & Re) === 0 && f.push(m), l = m, s = yt(m.next);
  }
  if (e.outrogroups !== null) {
    for (const B of e.outrogroups)
      B.pending.size === 0 && (Er(e, kr(B.done)), e.outrogroups?.delete(B));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var _ = [];
    if (u !== void 0)
      for (m of u)
        (m.f & ar) === 0 && _.push(m);
    for (; s !== null; )
      (s.f & ar) === 0 && s !== e.fallback && _.push(s), s = yt(s.next);
    var P = _.length;
    if (P > 0) {
      var I = null;
      Pa(e, _, I);
    }
  }
}
function Ba(e, t, r, n, i, a, o, s) {
  var u = (o & Ci) !== 0 ? (o & Ri) === 0 ? We(r, !1, !1) : Qr(r) : null, l = (o & Mi) !== 0 ? Qr(i) : null;
  return {
    v: u,
    i: l,
    e: Je(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function Et(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Re) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        Ui(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function Fe(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function wr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Oa(e, t, r) {
  var n = new Yt(e);
  At(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Xt);
}
const La = () => performance.now(), ye = {
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
function Vn() {
  const e = ye.now();
  ye.tasks.forEach((t) => {
    t.c(e) || (ye.tasks.delete(t), t.f());
  }), ye.tasks.size !== 0 && ye.tick(Vn);
}
function Na(e) {
  let t;
  return ye.tasks.size === 0 && ye.tick(Vn), {
    promise: new Promise((r) => {
      ye.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      ye.tasks.delete(t);
    }
  };
}
function Ma(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new Yt(s, !1);
  At(() => {
    const l = t() || null;
    var f = l === "svg" ? Fi : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (p) => {
      if (l) {
        if (o = On(l, f), st(o, o), n) {
          var v = null, E = o.appendChild(Ye());
          n(o, E), v?.remove();
        }
        zt.nodes.end = o, p.before(o);
      }
    }), () => {
    };
  }, Xt), Rr(() => {
  });
}
function Ca(e, t) {
  var r = void 0, n;
  Dn(() => {
    r !== (r = t()) && (n && (Ze(n), n = null), r && (n = Je(() => {
      Ur(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function zn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = zn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ra() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = zn(e)) && (n && (n += " "), n += t);
  return n;
}
function Da(e) {
  return typeof e == "object" ? Ra(e) : e ?? "";
}
const sn = [...` \t
\r\f \v\uFEFF`];
function ka(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || sn.includes(n[o - 1])) && (s === n.length || sn.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function on(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function sr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Ua(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(sr)), i && u.push(...Object.keys(i).map(sr));
      var l = 0, f = -1;
      const y = e.length;
      for (var p = 0; p < y; p++) {
        var v = e[p];
        if (s ? v === "/" && e[p - 1] === "*" && (s = !1) : a ? a === v && (a = !1) : v === "/" && e[p + 1] === "*" ? s = !0 : v === '"' || v === "'" ? a = v : v === "(" ? o++ : v === ")" && o--, !s && a === !1 && o === 0) {
          if (v === ":" && f === -1)
            f = p;
          else if (v === ";" || p === y - 1) {
            if (f !== -1) {
              var E = sr(e.substring(l, f).trim());
              if (!u.includes(E)) {
                v !== ";" && p++;
                var m = e.substring(l, p).trim();
                r += " " + m + ";";
              }
            }
            l = p + 1, f = -1;
          }
        }
      }
    }
    return n && (r += on(n)), i && (r += on(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function De(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[Kr]
  );
  if (o !== r || o === void 0) {
    var s = ka(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[Kr] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function or(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function xe(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[$r]
  );
  if (i !== t) {
    var a = Ua(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[$r] = t;
  } else n && (Array.isArray(n) ? (or(e, r?.[0], n[0]), or(e, r?.[1], n[1], "important")) : or(e, r, n));
  return n;
}
function Tr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Cr(t))
      return Gi();
    for (var n of e.options)
      n.selected = t.includes(ln(n));
    return;
  }
  for (n of e.options) {
    var i = ln(n);
    if (ji(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Fa(e) {
  var t = new MutationObserver(() => {
    Tr(e, e.__value);
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
  }), Rr(() => {
    t.disconnect();
  });
}
function ln(e) {
  return "__value" in e ? e.__value : e.value;
}
const wt = /* @__PURE__ */ Symbol("class"), nt = /* @__PURE__ */ Symbol("style"), Xn = /* @__PURE__ */ Symbol("is custom element"), qn = /* @__PURE__ */ Symbol("is html"), Ga = Fr ? "input" : "INPUT", ja = Fr ? "option" : "OPTION", Va = Fr ? "select" : "SELECT";
function za(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function at(e, t, r, n) {
  var i = Wn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Vi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Zn(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Xa(e, t, r, n, i = !1, a = !1) {
  var o = Wn(e), s = o[Xn], u = !o[qn], l = t || {}, f = e.nodeName === ja;
  for (var p in t)
    p in r || (r[p] = null);
  r.class ? r.class = Da(r.class) : r.class = null, r[nt] && (r.style ??= null);
  var v = Zn(e);
  if (e.nodeName === Ga && "type" in r && ("value" in r || "__value" in r)) {
    var E = r.type;
    (E !== l.type || E === void 0 && e.hasAttribute("type")) && (l.type = E, at(e, "type", E));
  }
  for (const g in r) {
    let _ = r[g];
    if (f && g === "value" && _ == null) {
      e.value = e.__value = "", l[g] = _;
      continue;
    }
    if (g === "class") {
      var m = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      De(e, m, _, n, t?.[wt], r[wt]), l[g] = _, l[wt] = r[wt];
      continue;
    }
    if (g === "style") {
      xe(e, _, t?.[nt], r[nt]), l[g] = _, l[nt] = r[nt];
      continue;
    }
    var y = l[g];
    if (!(_ === y && !(_ === void 0 && e.hasAttribute(g)))) {
      l[g] = _;
      var S = g[0] + g[1];
      if (S !== "$$")
        if (S === "on") {
          const P = {}, I = "$$" + g;
          let B = g.slice(2);
          var d = $i(B);
          if (Yi(B) && (B = B.slice(0, -7), P.capture = !0), !d && y) {
            if (_ != null) continue;
            e.removeEventListener(B, l[I], P), l[I] = null;
          }
          if (d)
            kn(B, e, _), qt([B]);
          else if (_ != null) {
            let N = function(R) {
              l[g].call(this, R);
            };
            l[I] = Ji(B, e, N, P);
          }
        } else if (g === "style")
          at(e, g, _);
        else if (g === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!_
          );
        else if (!s && (g === "__value" || g === "value" && _ != null))
          e.value = e.__value = _;
        else if (g === "selected" && f)
          za(
            /** @type {HTMLOptionElement} */
            e,
            _
          );
        else {
          var c = g;
          u || (c = Qi(c));
          var b = c === "defaultValue" || c === "defaultChecked";
          if (_ == null && !s && !b)
            if (o[g] = null, c === "value" || c === "checked") {
              let P = (
                /** @type {HTMLInputElement} */
                e
              );
              const I = t === void 0;
              if (c === "value") {
                let B = P.defaultValue;
                P.removeAttribute(c), P.defaultValue = B, P.value = P.__value = I ? B : null;
              } else {
                let B = P.defaultChecked;
                P.removeAttribute(c), P.defaultChecked = B, P.checked = I ? B : !1;
              }
            } else
              e.removeAttribute(g);
          else b || v.includes(c) && (s || typeof _ != "string") ? (e[c] = _, c in o && (o[c] = Ki)) : typeof _ != "function" && at(e, c, _);
        }
    }
  }
  return l;
}
function qa(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Wi(i, r, n, (u) => {
    var l = void 0, f = {}, p = e.nodeName === Va, v = !1;
    if (Dn(() => {
      var m = t(...u.map(h)), y = Xa(
        e,
        l,
        m,
        a,
        o,
        s
      );
      v && p && "value" in m && Tr(
        /** @type {HTMLSelectElement} */
        e,
        m.value
      );
      for (let d of Object.getOwnPropertySymbols(f))
        m[d] || Ze(f[d]);
      for (let d of Object.getOwnPropertySymbols(m)) {
        var S = m[d];
        d.description === Zi && (!l || S !== l[d]) && (f[d] && Ze(f[d]), f[d] = Je(() => Ca(e, () => S))), y[d] = S;
      }
      l = y;
    }), p) {
      var E = (
        /** @type {HTMLSelectElement} */
        e
      );
      Ur(() => {
        Tr(
          E,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), Fa(E);
      });
    }
    v = !0;
  });
}
function Wn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[zi] ??= {
      [Xn]: e.nodeName.includes("-"),
      [qn]: e.namespaceURI === Xi
    }
  );
}
var un = /* @__PURE__ */ new Map();
function Zn(e) {
  var t = e.getAttribute("is") || e.nodeName, r = un.get(t);
  if (r) return r;
  un.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = qi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Bn(i);
  }
  return r;
}
function lr(e, t) {
  return e === t || e?.[Gr] === t;
}
function Vr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Un.r
  ), a = (
    /** @type {Effect} */
    zt
  );
  return Ur(() => {
    var o, s;
    return ea(() => {
      o = s, s = [], te(() => {
        lr(r(...s), e) || (t(e, ...s), o && lr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & ta; )
        u = u.parent;
      const l = () => {
        s && lr(r(...s), e) && t(null, ...s);
      }, f = u.teardown;
      u.teardown = () => {
        l(), f?.();
      };
    };
  }), e;
}
function Wa(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Un
  ), r = t.l.u;
  if (!r) return;
  let n = () => me(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = _r(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => h(o);
  }
  r.b.length && ra(() => {
    fn(t, n), br(r.b);
  }), _e(() => {
    const i = te(() => r.m.map(na));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && _e(() => {
    fn(t, n), br(r.a);
  });
}
function fn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) h(r);
  t();
}
const Za = {
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
function Ya(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Za
  );
}
const Ja = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (_t(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      _t(i) && (i = i());
      const a = yr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (_t(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = yr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Gr || t === Fn) return !1;
    for (let r of e.props)
      if (_t(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (_t(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function Qa(...e) {
  return new Proxy({ props: e }, Ja);
}
function H(e, t, r, n) {
  var i = !oa || (r & la) !== 0, a = (r & sa) !== 0, o = (r & fa) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), f = () => o && i ? (l ??= _r(
    /** @type {() => V} */
    n
  ), h(l)) : (u && (u = !1, s = o ? te(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let p;
  if (a) {
    var v = Gr in e || Fn in e;
    p = yr(e, t)?.set ?? (v && t in e ? (g) => e[t] = g : void 0);
  }
  var E, m = !1;
  a ? [E, m] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : E = /** @type {V} */
  e[t], E === void 0 && n !== void 0 && (E = f(), p && (i && ia(), p(E)));
  var y;
  if (i ? y = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g === void 0 ? f() : (u = !0, g);
  } : y = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g !== void 0 && (s = /** @type {V} */
    void 0), g === void 0 ? s : g;
  }, i && (r & aa) === 0)
    return y;
  if (p) {
    var S = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(g, _) {
        return arguments.length > 0 ? ((!i || !_ || S || m) && p(_ ? y() : g), g) : y();
      })
    );
  }
  var d = !1, c = ((r & ua) !== 0 ? _r : Cn)(() => (d = !1, y()));
  a && h(c);
  var b = (
    /** @type {Effect} */
    zt
  );
  return (
    /** @type {() => V} */
    (function(g, _) {
      if (arguments.length > 0) {
        const P = _ ? h(c) : i && a ? Tt(g) : g;
        return w(c, P), d = !0, s !== void 0 && (s = P), g;
      }
      return ha && d || (b.f & Rn) !== 0 ? c.v : h(c);
    })
  );
}
ca();
var Ka = /* @__PURE__ */ jn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), hn = /* @__PURE__ */ re("<!> <!>", 1), $a = /* @__PURE__ */ re('<div class="placeholder svelte-1stq1b1"></div>');
function es(e, t) {
  Zt(t, !1);
  let r = H(t, "height", 8, void 0), n = H(t, "min_height", 8, void 0), i = H(t, "max_height", 8, void 0), a = H(t, "width", 8, void 0), o = H(t, "elem_id", 8, ""), s = H(t, "elem_classes", 24, () => []), u = H(t, "variant", 8, "solid"), l = H(t, "border_mode", 8, "base"), f = H(t, "padding", 8, !0), p = H(t, "type", 8, "normal"), v = H(t, "test_id", 8, void 0), E = H(t, "explicit_call", 8, !1), m = H(t, "container", 8, !0), y = H(t, "visible", 8, !0), S = H(t, "allow_overflow", 8, !0), d = H(t, "overflow_behavior", 8, "auto"), c = H(t, "scale", 8, null), b = H(t, "min_width", 8, 0), g = H(t, "flex", 12, !1), _ = H(t, "resizable", 8, !1), P = H(t, "rtl", 8, !1), I = H(t, "fullscreen", 12, !1), B = H(t, "label", 8, void 0), N = We(I()), R = We(), $ = p() === "fieldset" ? "fieldset" : "div", ge = We(0), ne = We(0), V = We(null);
  function le(J) {
    I() && J.key === "Escape" && I(!1);
  }
  const ce = (J) => {
    if (J !== void 0) {
      if (typeof J == "number")
        return J + "px";
      if (typeof J == "string")
        return J;
    }
  }, Ve = (J) => {
    let we = J.clientY;
    const pe = (ee) => {
      const ue = ee.clientY - we;
      we = ee.clientY, pa(R, h(R).style.height = `${h(R).offsetHeight + ue}px`);
    }, Ie = () => {
      window.removeEventListener("mousemove", pe), window.removeEventListener("mouseup", Ie);
    };
    window.addEventListener("mousemove", pe), window.addEventListener("mouseup", Ie);
  };
  en(
    () => (me(I()), h(N), h(R)),
    () => {
      I() !== h(N) && (w(N, I()), I() ? (w(V, h(R).getBoundingClientRect()), w(ge, h(R).offsetHeight), w(ne, h(R).offsetWidth), window.addEventListener("keydown", le)) : (w(V, null), window.removeEventListener("keydown", le)));
    }
  ), en(() => me(y()), () => {
    y() || g(!1);
  }), da(), Wa();
  var Ee = it(), de = se(Ee);
  {
    var dt = (J) => {
      var we = hn(), pe = se(we);
      Ma(pe, () => $, !1, (ue, Te) => {
        Vr(ue, (ve) => w(R, ve), () => h(R)), qa(
          ue,
          (ve, Oe) => ({
            "data-testid": v(),
            id: o(),
            class: `block ${ve ?? ""}`,
            dir: P() ? "rtl" : "ltr",
            "aria-label": B(),
            style: "",
            [wt]: {
              hidden: y() === "hidden",
              padded: f(),
              flex: g(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !E() && !m(),
              fullscreen: I(),
              animating: I() && h(V) !== null,
              "auto-margin": c() === null
            },
            [nt]: Oe
          }),
          [
            () => (me(s()), te(() => s()?.join(" ") || "")),
            () => ({
              height: (me(I()), me(r()), te(() => I() ? void 0 : ce(r()))),
              "min-height": (me(I()), me(n()), te(() => I() ? void 0 : ce(n()))),
              "max-height": (me(I()), me(i()), te(() => I() ? void 0 : ce(i()))),
              "--start-top": (h(V), te(() => h(V) ? `${h(V).top}px` : "0px")),
              "--start-left": (h(V), te(() => h(V) ? `${h(V).left}px` : "0px")),
              "--start-width": (h(V), te(() => h(V) ? `${h(V).width}px` : "0px")),
              "--start-height": (h(V), te(() => h(V) ? `${h(V).height}px` : "0px")),
              width: (me(I()), me(a()), te(() => I() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : ce(a()))),
              "border-style": u(),
              overflow: S() ? d() : "hidden",
              "flex-grow": c(),
              "min-width": `calc(min(${b()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Se = hn(), ze = se(Se);
        wr(ze, t, "default", {});
        var Be = j(ze, 2);
        {
          var Ke = (ve) => {
            var Oe = Ka();
            Ce("mousedown", Oe, Ve), C(ve, Oe);
          };
          W(Be, (ve) => {
            _() && ve(Ke);
          });
        }
        C(Te, Se);
      });
      var Ie = j(pe, 2);
      {
        var ee = (ue) => {
          var Te = $a();
          let Se;
          Y(() => Se = xe(Te, "", Se, {
            height: h(ge) + "px",
            width: h(ne) + "px"
          })), C(ue, Te);
        };
        W(Ie, (ue) => {
          I() && ue(ee);
        });
      }
      C(J, we);
    };
    W(de, (J) => {
      (y() === !0 || y() === "hidden") && J(dt);
    });
  }
  C(e, Ee), Wt();
}
var ts = /* @__PURE__ */ re('<span class="svelte-vvirtv"> </span>'), rs = /* @__PURE__ */ re("<button><!> <div><!> <!></div></button>");
function cn(e, t) {
  let r = H(t, "label", 3, ""), n = H(t, "show_label", 3, !1), i = H(t, "pending", 3, !1), a = H(t, "size", 3, "small"), o = H(t, "padded", 3, !0), s = H(t, "highlight", 3, !1), u = H(t, "disabled", 3, !1), l = H(t, "hasPopup", 3, !1), f = H(t, "color", 3, "var(--block-label-text-color)"), p = H(t, "transparent", 3, !1), v = H(t, "background", 3, "var(--block-background-fill)"), E = H(t, "border", 3, "transparent"), m = be(() => s() ? "var(--color-accent)" : f());
  var y = rs();
  let S, d;
  var c = K(y);
  {
    var b = (N) => {
      var R = ts(), $ = K(R);
      Y(() => ae($, r())), C(N, R);
    };
    W(c, (N) => {
      n() && N(b);
    });
  }
  var g = j(c, 2);
  let _;
  var P = K(g);
  Oa(P, () => t.Icon, (N, R) => {
    R(N, {});
  });
  var I = j(P, 2);
  {
    var B = (N) => {
      var R = it(), $ = se(R);
      Ha($, () => t.children), C(N, R);
    };
    W(I, (N) => {
      t.children && N(B);
    });
  }
  Y(() => {
    S = De(y, 1, "icon-button svelte-vvirtv", null, S, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: p()
    }), y.disabled = u(), at(y, "aria-label", r()), at(y, "aria-haspopup", l()), at(y, "title", r()), d = xe(y, "", d, {
      "--border-color": E(),
      color: !u() && h(m) ? h(m) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : v()
    }), _ = De(g, 1, "svelte-vvirtv", null, _, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), kn("click", y, function(...N) {
    t.onclick?.apply(this, N);
  }), C(e, y);
}
qt(["click"]);
var ns = /* @__PURE__ */ jn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function dn(e) {
  var t = ns();
  C(e, t);
}
const is = [
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
], pn = {
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
is.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: pn[t][r],
    secondary: pn[t][n]
  }
}), {});
function as(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var ur, vn;
function ss() {
  if (vn) return ur;
  vn = 1;
  var e = function(c) {
    return t(c) && !r(c);
  };
  function t(d) {
    return !!d && typeof d == "object";
  }
  function r(d) {
    var c = Object.prototype.toString.call(d);
    return c === "[object RegExp]" || c === "[object Date]" || a(d);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(d) {
    return d.$$typeof === i;
  }
  function o(d) {
    return Array.isArray(d) ? [] : {};
  }
  function s(d, c) {
    return c.clone !== !1 && c.isMergeableObject(d) ? y(o(d), d, c) : d;
  }
  function u(d, c, b) {
    return d.concat(c).map(function(g) {
      return s(g, b);
    });
  }
  function l(d, c) {
    if (!c.customMerge)
      return y;
    var b = c.customMerge(d);
    return typeof b == "function" ? b : y;
  }
  function f(d) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(d).filter(function(c) {
      return Object.propertyIsEnumerable.call(d, c);
    }) : [];
  }
  function p(d) {
    return Object.keys(d).concat(f(d));
  }
  function v(d, c) {
    try {
      return c in d;
    } catch {
      return !1;
    }
  }
  function E(d, c) {
    return v(d, c) && !(Object.hasOwnProperty.call(d, c) && Object.propertyIsEnumerable.call(d, c));
  }
  function m(d, c, b) {
    var g = {};
    return b.isMergeableObject(d) && p(d).forEach(function(_) {
      g[_] = s(d[_], b);
    }), p(c).forEach(function(_) {
      E(d, _) || (v(d, _) && b.isMergeableObject(c[_]) ? g[_] = l(_, b)(d[_], c[_], b) : g[_] = s(c[_], b));
    }), g;
  }
  function y(d, c, b) {
    b = b || {}, b.arrayMerge = b.arrayMerge || u, b.isMergeableObject = b.isMergeableObject || e, b.cloneUnlessOtherwiseSpecified = s;
    var g = Array.isArray(c), _ = Array.isArray(d), P = g === _;
    return P ? g ? b.arrayMerge(d, c, b) : m(d, c, b) : s(c, b);
  }
  y.all = function(c, b) {
    if (!Array.isArray(c))
      throw new Error("first argument should be an array");
    return c.reduce(function(g, _) {
      return y(g, _, b);
    }, {});
  };
  var S = y;
  return ur = S, ur;
}
var os = ss();
const ls = /* @__PURE__ */ as(os);
var Sr = function(e, t) {
  return Sr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Sr(e, t);
};
function Jt(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Sr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var U = function() {
  return U = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, U.apply(this, arguments);
};
function us(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function fr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function hr(e, t) {
  var r = t && t.cache ? t.cache : ms, n = t && t.serializer ? t.serializer : ps, i = t && t.strategy ? t.strategy : cs;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function fs(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function hs(e, t, r, n) {
  var i = fs(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function Yn(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function Jn(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function cs(e, t) {
  var r = e.length === 1 ? hs : Yn;
  return Jn(e, this, r, t.cache.create(), t.serializer);
}
function ds(e, t) {
  return Jn(e, this, Yn, t.cache.create(), t.serializer);
}
var ps = function() {
  return JSON.stringify(arguments);
}, vs = (
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
), ms = {
  create: function() {
    return new vs();
  }
}, cr = {
  variadic: ds
}, M;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(M || (M = {}));
var z;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(z || (z = {}));
var ot;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(ot || (ot = {}));
function mn(e) {
  return e.type === z.literal;
}
function gs(e) {
  return e.type === z.argument;
}
function Qn(e) {
  return e.type === z.number;
}
function Kn(e) {
  return e.type === z.date;
}
function $n(e) {
  return e.type === z.time;
}
function ei(e) {
  return e.type === z.select;
}
function ti(e) {
  return e.type === z.plural;
}
function bs(e) {
  return e.type === z.pound;
}
function ri(e) {
  return e.type === z.tag;
}
function ni(e) {
  return !!(e && typeof e == "object" && e.type === ot.number);
}
function Ar(e) {
  return !!(e && typeof e == "object" && e.type === ot.dateTime);
}
var ii = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, _s = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function ys(e) {
  var t = {};
  return e.replace(_s, function(r) {
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
var xs = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Es(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(xs).filter(function(v) {
    return v.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, f = u; l < f.length; l++) {
      var p = f[l];
      if (p.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function ws(e) {
  return e.replace(/^(.*?)-/, "");
}
var gn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, ai = /^(@+)?(\+|#+)?[rs]?$/g, Ts = /(\*)(0+)|(#+)(0+)|(0+)/g, si = /^(0+)$/;
function bn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(ai, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function oi(e) {
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
function Ss(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !si.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function _n(e) {
  var t = {}, r = oi(e);
  return r || t;
}
function As(e) {
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
        t.style = "unit", t.unit = ws(i.options[0]);
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
        t = U(U(U({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return U(U({}, u), _n(l));
        }, {}));
        continue;
      case "engineering":
        t = U(U(U({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return U(U({}, u), _n(l));
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
        i.options[0].replace(Ts, function(u, l, f, p, v, E) {
          if (l)
            t.minimumIntegerDigits = f.length;
          else {
            if (p && v)
              throw new Error("We currently do not support maximum integer digits");
            if (E)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (si.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (gn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(gn, function(u, l, f, p, v, E) {
        return f === "*" ? t.minimumFractionDigits = l.length : p && p[0] === "#" ? t.maximumFractionDigits = p.length : v && E ? (t.minimumFractionDigits = v.length, t.maximumFractionDigits = v.length + E.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = U(U({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = U(U({}, t), bn(a)));
      continue;
    }
    if (ai.test(i.stem)) {
      t = U(U({}, t), bn(i.stem));
      continue;
    }
    var o = oi(i.stem);
    o && (t = U(U({}, t), o));
    var s = Ss(i.stem);
    s && (t = U(U({}, t), s));
  }
  return t;
}
var Ct = {
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
function Hs(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), u = "a", l = Ps(t);
      for ((l == "H" || l == "k") && (s = 0); s-- > 0; )
        r += u;
      for (; o-- > 0; )
        r = l + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Ps(e) {
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
  var i = Ct[n || ""] || Ct[r || ""] || Ct["".concat(r, "-001")] || Ct["001"];
  return i[0];
}
var dr, Is = new RegExp("^".concat(ii.source, "*")), Bs = new RegExp("".concat(ii.source, "*$"));
function D(e, t) {
  return { start: e, end: t };
}
var Os = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ls = !!String.fromCodePoint, Ns = !!Object.fromEntries, Ms = !!String.prototype.codePointAt, Cs = !!String.prototype.trimStart, Rs = !!String.prototype.trimEnd, Ds = !!Number.isSafeInteger, ks = Ds ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Hr = !0;
try {
  var Us = ui("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Hr = ((dr = Us.exec("a")) === null || dr === void 0 ? void 0 : dr[0]) === "a";
} catch {
  Hr = !1;
}
var yn = Os ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Pr = Ls ? String.fromCodePoint : (
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
  Ns ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), li = Ms ? (
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
), Fs = Cs ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Is, "");
  }
), Gs = Rs ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Bs, "");
  }
);
function ui(e, t) {
  return new RegExp(e, t);
}
var Ir;
if (Hr) {
  var En = ui("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Ir = function(t, r) {
    var n;
    En.lastIndex = r;
    var i = En.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Ir = function(t, r) {
    for (var n = []; ; ) {
      var i = li(t, r);
      if (i === void 0 || fi(i) || Xs(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Pr.apply(void 0, n);
  };
var js = (
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
              type: z.pound,
              location: D(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(M.UNMATCHED_CLOSING_TAG, D(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Br(this.peek() || 0)) {
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
            type: z.literal,
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
          if (this.isEOF() || !Br(this.char()))
            return this.error(M.INVALID_TAG, D(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(M.UNMATCHED_CLOSING_TAG, D(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: z.tag,
              value: i,
              children: o,
              location: D(n, this.clonePosition())
            },
            err: null
          } : this.error(M.INVALID_TAG, D(s, this.clonePosition())));
        } else
          return this.error(M.UNCLOSED_TAG, D(n, this.clonePosition()));
      } else
        return this.error(M.INVALID_TAG, D(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && zs(this.char()); )
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
        val: { type: z.literal, value: i, location: u },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Vs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Pr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Pr(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(M.EMPTY_ARGUMENT, D(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(M.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: z.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: D(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(M.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Ir(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = D(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(M.EXPECT_ARGUMENT_TYPE, D(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var f = this.clonePosition(), p = this.parseSimpleArgStyleIfPossible();
            if (p.err)
              return p;
            var v = Gs(p.val);
            if (v.length === 0)
              return this.error(M.EXPECT_ARGUMENT_STYLE, D(this.clonePosition(), this.clonePosition()));
            var E = D(f, this.clonePosition());
            l = { style: v, styleLocation: E };
          }
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var y = D(i, this.clonePosition());
          if (l && yn(l?.style, "::", 0)) {
            var S = Fs(l.style.slice(2));
            if (s === "number") {
              var p = this.parseNumberSkeletonFromString(S, l.styleLocation);
              return p.err ? p : {
                val: { type: z.number, value: n, location: y, style: p.val },
                err: null
              };
            } else {
              if (S.length === 0)
                return this.error(M.EXPECT_DATE_TIME_SKELETON, y);
              var d = S;
              this.locale && (d = Hs(S, this.locale));
              var v = {
                type: ot.dateTime,
                pattern: d,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? ys(d) : {}
              }, c = s === "date" ? z.date : z.time;
              return {
                val: { type: c, value: n, location: y, style: v },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? z.number : s === "date" ? z.date : z.time,
              value: n,
              location: y,
              style: (a = l?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var b = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(M.EXPECT_SELECT_ARGUMENT_OPTIONS, D(b, U({}, b)));
          this.bumpSpace();
          var g = this.parseIdentifierIfPossible(), _ = 0;
          if (s !== "select" && g.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(M.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, D(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var p = this.tryParseDecimalInteger(M.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, M.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (p.err)
              return p;
            this.bumpSpace(), g = this.parseIdentifierIfPossible(), _ = p.val;
          }
          var P = this.tryParsePluralOrSelectOptions(t, s, r, g);
          if (P.err)
            return P;
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var I = D(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: z.select,
              value: n,
              options: xn(P.val),
              location: I
            },
            err: null
          } : {
            val: {
              type: z.plural,
              value: n,
              options: xn(P.val),
              offset: _,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: I
            },
            err: null
          };
        }
        default:
          return this.error(M.INVALID_ARGUMENT_TYPE, D(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, D(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(M.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, D(i, this.clonePosition()));
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
        n = Es(t);
      } catch {
        return this.error(M.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: ot.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? As(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, f = i.location; ; ) {
        if (l.length === 0) {
          var p = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var v = this.tryParseDecimalInteger(M.EXPECT_PLURAL_ARGUMENT_SELECTOR, M.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (v.err)
              return v;
            f = D(p, this.clonePosition()), l = this.message.slice(p.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? M.DUPLICATE_SELECT_ARGUMENT_SELECTOR : M.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, f);
        l === "other" && (o = !0), this.bumpSpace();
        var E = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? M.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : M.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, D(this.clonePosition(), this.clonePosition()));
        var m = this.parseMessage(t + 1, r, n);
        if (m.err)
          return m;
        var y = this.tryParseArgumentClose(E);
        if (y.err)
          return y;
        s.push([
          l,
          {
            value: m.val,
            location: D(E, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, f = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? M.EXPECT_SELECT_ARGUMENT_SELECTOR : M.EXPECT_PLURAL_ARGUMENT_SELECTOR, D(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(M.MISSING_OTHER_CLAUSE, D(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      return a ? (o *= n, ks(o) ? { val: o, err: null } : this.error(r, u)) : this.error(t, u);
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
      var r = li(this.message, t);
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
      for (; !this.isEOF() && fi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Br(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Vs(e) {
  return Br(e) || e === 47;
}
function zs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function fi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Xs(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Or(e) {
  e.forEach(function(t) {
    if (delete t.location, ei(t) || ti(t))
      for (var r in t.options)
        delete t.options[r].location, Or(t.options[r].value);
    else Qn(t) && ni(t.style) || (Kn(t) || $n(t)) && Ar(t.style) ? delete t.style.location : ri(t) && Or(t.children);
  });
}
function qs(e, t) {
  t === void 0 && (t = {}), t = U({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new js(e, t).parse();
  if (r.err) {
    var n = SyntaxError(M[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Or(r.val), r.val;
}
var lt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(lt || (lt = {}));
var Qt = (
  /** @class */
  (function(e) {
    Jt(t, e);
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
    Jt(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), lt.INVALID_VALUE, a) || this;
    }
    return t;
  })(Qt)
), Ws = (
  /** @class */
  (function(e) {
    Jt(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), lt.INVALID_VALUE, i) || this;
    }
    return t;
  })(Qt)
), Zs = (
  /** @class */
  (function(e) {
    Jt(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), lt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Qt)
), oe;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(oe || (oe = {}));
function Ys(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== oe.literal || r.type !== oe.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Js(e) {
  return typeof e == "function";
}
function kt(e, t, r, n, i, a, o) {
  if (e.length === 1 && mn(e[0]))
    return [
      {
        type: oe.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var f = l[u];
    if (mn(f)) {
      s.push({
        type: oe.literal,
        value: f.value
      });
      continue;
    }
    if (bs(f)) {
      typeof a == "number" && s.push({
        type: oe.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var p = f.value;
    if (!(i && p in i))
      throw new Zs(p, o);
    var v = i[p];
    if (gs(f)) {
      (!v || typeof v == "string" || typeof v == "number") && (v = typeof v == "string" || typeof v == "number" ? String(v) : ""), s.push({
        type: typeof v == "string" ? oe.literal : oe.object,
        value: v
      });
      continue;
    }
    if (Kn(f)) {
      var E = typeof f.style == "string" ? n.date[f.style] : Ar(f.style) ? f.style.parsedOptions : void 0;
      s.push({
        type: oe.literal,
        value: r.getDateTimeFormat(t, E).format(v)
      });
      continue;
    }
    if ($n(f)) {
      var E = typeof f.style == "string" ? n.time[f.style] : Ar(f.style) ? f.style.parsedOptions : n.time.medium;
      s.push({
        type: oe.literal,
        value: r.getDateTimeFormat(t, E).format(v)
      });
      continue;
    }
    if (Qn(f)) {
      var E = typeof f.style == "string" ? n.number[f.style] : ni(f.style) ? f.style.parsedOptions : void 0;
      E && E.scale && (v = v * (E.scale || 1)), s.push({
        type: oe.literal,
        value: r.getNumberFormat(t, E).format(v)
      });
      continue;
    }
    if (ri(f)) {
      var m = f.children, y = f.value, S = i[y];
      if (!Js(S))
        throw new Ws(y, "function", o);
      var d = kt(m, t, r, n, i, a), c = S(d.map(function(_) {
        return _.value;
      }));
      Array.isArray(c) || (c = [c]), s.push.apply(s, c.map(function(_) {
        return {
          type: typeof _ == "string" ? oe.literal : oe.object,
          value: _
        };
      }));
    }
    if (ei(f)) {
      var b = f.options[v] || f.options.other;
      if (!b)
        throw new wn(f.value, v, Object.keys(f.options), o);
      s.push.apply(s, kt(b.value, t, r, n, i));
      continue;
    }
    if (ti(f)) {
      var b = f.options["=".concat(v)];
      if (!b) {
        if (!Intl.PluralRules)
          throw new Qt(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, lt.MISSING_INTL_API, o);
        var g = r.getPluralRules(t, { type: f.pluralType }).select(v - (f.offset || 0));
        b = f.options[g] || f.options.other;
      }
      if (!b)
        throw new wn(f.value, v, Object.keys(f.options), o);
      s.push.apply(s, kt(b.value, t, r, n, i, v - (f.offset || 0)));
      continue;
    }
  }
  return Ys(s);
}
function Qs(e, t) {
  return t ? U(U(U({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = U(U({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ks(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Qs(e[n], t[n]), r;
  }, U({}, e)) : e;
}
function pr(e) {
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
function $s(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: hr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, fr([void 0], r, !1)))();
    }, {
      cache: pr(e.number),
      strategy: cr.variadic
    }),
    getDateTimeFormat: hr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, fr([void 0], r, !1)))();
    }, {
      cache: pr(e.dateTime),
      strategy: cr.variadic
    }),
    getPluralRules: hr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, fr([void 0], r, !1)))();
    }, {
      cache: pr(e.pluralRules),
      strategy: cr.variadic
    })
  };
}
var eo = (
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
        var f = l.reduce(function(p, v) {
          return !p.length || v.type !== oe.literal || typeof p[p.length - 1] != "string" ? p.push(v.value) : p[p.length - 1] += v.value, p;
        }, []);
        return f.length <= 1 ? f[0] || "" : f;
      }, this.formatToParts = function(u) {
        return kt(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
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
        var s = us(o, ["formatters"]);
        this.ast = e.__parse(t, U(U({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Ks(e.formats, n), this.formatters = i && i.formatters || $s(this.formatterCache);
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
    }, e.__parse = qs, e.formats = {
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
function to(e, t) {
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
const Ge = {}, ro = (e, t, r) => r && (t in Ge || (Ge[t] = {}), e in Ge[t] || (Ge[t][e] = r), r), hi = (e, t) => {
  if (t == null)
    return;
  if (t in Ge && e in Ge[t])
    return Ge[t][e];
  const r = Kt(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = io(i, e);
    if (a)
      return ro(e, t, a);
  }
};
let zr;
const Pt = Ht({});
function no(e) {
  return zr[e] || null;
}
function ci(e) {
  return e in zr;
}
function io(e, t) {
  if (!ci(e))
    return null;
  const r = no(e);
  return to(r, t);
}
function ao(e) {
  if (e == null)
    return;
  const t = Kt(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (ci(n))
      return n;
  }
}
function so(e, ...t) {
  delete Ge[e], Pt.update((r) => (r[e] = ls.all([r[e] || {}, ...t]), r));
}
ft(
  [Pt],
  ([e]) => Object.keys(e)
);
Pt.subscribe((e) => zr = e);
const Ut = {};
function oo(e, t) {
  Ut[e].delete(t), Ut[e].size === 0 && delete Ut[e];
}
function di(e) {
  return Ut[e];
}
function lo(e) {
  return Kt(e).map((t) => {
    const r = di(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Lr(e) {
  return e == null ? !1 : Kt(e).some(
    (t) => {
      var r;
      return (r = di(t)) == null ? void 0 : r.size;
    }
  );
}
function uo(e, t) {
  return Promise.all(
    t.map((n) => (oo(e, n), n().then((i) => i.default || i)))
  ).then((n) => so(e, ...n));
}
const xt = {};
function pi(e) {
  if (!Lr(e))
    return e in xt ? xt[e] : Promise.resolve();
  const t = lo(e);
  return xt[e] = Promise.all(
    t.map(
      ([r, n]) => uo(r, n)
    )
  ).then(() => {
    if (Lr(e))
      return pi(e);
    delete xt[e];
  }), xt[e];
}
const fo = {
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
}, ho = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: fo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, co = ho;
function ut() {
  return co;
}
const vr = Ht(!1);
var po = Object.defineProperty, vo = Object.defineProperties, mo = Object.getOwnPropertyDescriptors, Tn = Object.getOwnPropertySymbols, go = Object.prototype.hasOwnProperty, bo = Object.prototype.propertyIsEnumerable, Sn = (e, t, r) => t in e ? po(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, _o = (e, t) => {
  for (var r in t || (t = {}))
    go.call(t, r) && Sn(e, r, t[r]);
  if (Tn)
    for (var r of Tn(t))
      bo.call(t, r) && Sn(e, r, t[r]);
  return e;
}, yo = (e, t) => vo(e, mo(t));
let Nr;
const jt = Ht(null);
function An(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Kt(e, t = ut().fallbackLocale) {
  const r = An(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...An(t)])] : r;
}
function Qe() {
  return Nr ?? void 0;
}
jt.subscribe((e) => {
  Nr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const xo = (e) => {
  if (e && ao(e) && Lr(e)) {
    const { loadingDelay: t } = ut();
    let r;
    return typeof window < "u" && Qe() != null && t ? r = window.setTimeout(
      () => vr.set(!0),
      t
    ) : vr.set(!0), pi(e).then(() => {
      jt.set(e);
    }).finally(() => {
      clearTimeout(r), vr.set(!1);
    });
  }
  return jt.set(e);
}, ht = yo(_o({}, jt), {
  set: xo
}), $t = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Eo = Object.defineProperty, Vt = Object.getOwnPropertySymbols, vi = Object.prototype.hasOwnProperty, mi = Object.prototype.propertyIsEnumerable, Hn = (e, t, r) => t in e ? Eo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Xr = (e, t) => {
  for (var r in t || (t = {}))
    vi.call(t, r) && Hn(e, r, t[r]);
  if (Vt)
    for (var r of Vt(t))
      mi.call(t, r) && Hn(e, r, t[r]);
  return e;
}, ct = (e, t) => {
  var r = {};
  for (var n in e)
    vi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Vt)
    for (var n of Vt(e))
      t.indexOf(n) < 0 && mi.call(e, n) && (r[n] = e[n]);
  return r;
};
const St = (e, t) => {
  const { formats: r } = ut();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, wo = $t(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = St("number", n)), new Intl.NumberFormat(r, i);
  }
), To = $t(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = St("date", n) : Object.keys(i).length === 0 && (i = St("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), So = $t(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = St("time", n) : Object.keys(i).length === 0 && (i = St("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Ao = (e = {}) => {
  var t = e, {
    locale: r = Qe()
  } = t, n = ct(t, [
    "locale"
  ]);
  return wo(Xr({ locale: r }, n));
}, Ho = (e = {}) => {
  var t = e, {
    locale: r = Qe()
  } = t, n = ct(t, [
    "locale"
  ]);
  return To(Xr({ locale: r }, n));
}, Po = (e = {}) => {
  var t = e, {
    locale: r = Qe()
  } = t, n = ct(t, [
    "locale"
  ]);
  return So(Xr({ locale: r }, n));
}, Io = $t(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Qe()) => new eo(e, t, ut().formats, {
    ignoreTag: ut().ignoreTag
  })
), Bo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = Qe(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let f = hi(e, u);
  if (!f)
    f = (a = (i = (n = (r = ut()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof f != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof f}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), f;
  if (!s)
    return f;
  let p = f;
  try {
    p = Io(f, u).format(s);
  } catch (v) {
    v instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      v.message
    );
  }
  return p;
}, Oo = (e, t) => Po(t).format(e), Lo = (e, t) => Ho(t).format(e), No = (e, t) => Ao(t).format(e), Mo = (e, t = Qe()) => hi(e, t);
ft([ht, Pt], () => Bo);
ft([ht], () => Oo);
ft([ht], () => Lo);
ft([ht], () => No);
ft([ht, Pt], () => Mo);
const Co = "__i18n__", Ro = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Do = [
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
function ko(e) {
  return typeof e == "string" && e.includes(Co);
}
class Uo {
  load_component;
  #t = q(Tt({}));
  get shared() {
    return h(this.#t);
  }
  set shared(t) {
    w(this.#t, t, !0);
  }
  #r = q(Tt({}));
  get props() {
    return h(this.#r);
  }
  set props(t) {
    w(this.#r, t, !0);
  }
  #e = q((t) => t);
  get i18n() {
    return h(this.#e);
  }
  set i18n(t) {
    w(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Do;
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
    for (const n of Ro)
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
    ), _e(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), te(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && ht.subscribe(() => {
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
      const n = t[r], i = ko(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    _e(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
qt(["click"]);
function mr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Pn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Mr(e, t, r, n) {
  if (typeof r == "number" || Pn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Pn(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Mr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Mr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function In(e, t = {}) {
  const r = Ht(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), f = (
    /** @type {T | undefined} */
    e
  ), p = 1, v = 0, E = !1;
  function m(S, d = {}) {
    f = S;
    const c = u = {};
    return e == null || d.hard || y.stiffness >= 1 && y.damping >= 1 ? (E = !0, o = ye.now(), l = S, r.set(e = f), Promise.resolve()) : (d.soft && (v = 1 / ((d.soft === !0 ? 0.5 : +d.soft) * 60), p = 0), s || (o = ye.now(), E = !1, s = Na((b) => {
      if (E)
        return E = !1, s = null, !1;
      p = Math.min(p + v, 1);
      const g = Math.min(b - o, 1e3 / 30), _ = {
        inv_mass: p,
        opts: y,
        settled: !0,
        dt: g * 60 / 1e3
      }, P = Mr(_, l, e, f);
      return o = b, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      P), _.settled && (s = null), !_.settled;
    })), new Promise((b) => {
      s.promise.then(() => {
        c === u && b();
      });
    }));
  }
  const y = {
    set: m,
    update: (S, d) => m(S(
      /** @type {T} */
      f,
      /** @type {T} */
      e
    ), d),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return y;
}
var Fo = /* @__PURE__ */ re('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function Go(e, t) {
  Zt(t, !0);
  const r = () => tn(u, "$top", i), n = () => tn(l, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(b, g, _, P) {
    function I(B) {
      return B instanceof _ ? B : new _(function(N) {
        N(B);
      });
    }
    return new (_ || (_ = Promise))(function(B, N) {
      function R(ne) {
        try {
          ge(P.next(ne));
        } catch (V) {
          N(V);
        }
      }
      function $(ne) {
        try {
          ge(P.throw(ne));
        } catch (V) {
          N(V);
        }
      }
      function ge(ne) {
        ne.done ? B(ne.value) : I(ne.value).then(R, $);
      }
      ge((P = P.apply(b, g || [])).next());
    });
  };
  let s = H(t, "margin", 3, !0);
  const u = In([0, 0]), l = In([0, 0]);
  let f = q(!1);
  function p() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function v() {
    return o(this, void 0, void 0, function* () {
      yield p(), h(f) || v();
    });
  }
  function E() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), v();
    });
  }
  _e(() => (E(), () => {
    w(f, !0);
  }));
  var m = Fo();
  let y;
  var S = K(m), d = K(S), c = j(d);
  Y(() => {
    y = De(m, 1, "svelte-m6d381", null, y, { margin: s() }), xe(d, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), xe(c, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), C(e, m), Wt(), a();
}
var jo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(f) {
      try {
        l(n.next(f));
      } catch (p) {
        o(p);
      }
    }
    function u(f) {
      try {
        l(n.throw(f));
      } catch (p) {
        o(p);
      }
    }
    function l(f) {
      f.done ? a(f.value) : i(f.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let Rt = [], gr = !1;
const Vo = typeof window < "u", gi = Vo ? window.requestAnimationFrame : (e) => {
};
function zo(e) {
  return jo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Rt.push(t), !gr) gr = !0;
      else return;
      yield va(), gi(() => {
        let n = [0, 0];
        for (let i = 0; i < Rt.length; i++) {
          const o = Rt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), gr = !1, Rt = [];
      });
    }
  });
}
var Xo = /* @__PURE__ */ re('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), qo = /* @__PURE__ */ re('<div class="eta-bar svelte-124hqw6"></div>'), Wo = /* @__PURE__ */ re("<!> ", 1), Zo = /* @__PURE__ */ re("<!> <!> <!> <!>", 1), Yo = /* @__PURE__ */ re('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), Jo = /* @__PURE__ */ re('<p class="loading svelte-124hqw6"> </p> <!>', 1), Qo = /* @__PURE__ */ re("<!> <div><!> <!></div> <!> <!>", 1), Ko = /* @__PURE__ */ re('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), $o = /* @__PURE__ */ re("<div> <!> </div>"), el = /* @__PURE__ */ re('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function tl(e, t) {
  Zt(t, !0);
  let r = H(t, "eta", 3, null), n = H(t, "scroll_to_output", 3, !1), i = H(t, "timer", 3, !0), a = H(t, "show_progress", 3, "full"), o = H(t, "message", 3, null), s = H(t, "progress", 3, null), u = H(t, "variant", 3, "default"), l = H(t, "loading_text", 3, "Loading..."), f = H(t, "absolute", 3, !0), p = H(t, "translucent", 3, !1), v = H(t, "border", 3, !1), E = H(t, "validation_error", 7, null), m = H(t, "show_validation_error", 3, !0), y = H(t, "type", 3, null), S = H(t, "used_cache", 3, null), d = H(t, "cache_duration", 3, null), c = H(t, "avg_time", 3, null), b, g = !1, _ = q(0), P = q(null), I = q(null), B = q(!1), N = q(null), R = q(!1), $ = q(!1), ge = q(null), ne = q(null), V = q("from cache"), le = q(!1), ce = null, Ve = null;
  const Ee = be(() => !(m() && E()) && (y() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let de = q(0);
  const dt = be(() => h(I) === null || h(I) <= 0 || !h(de) ? 0 : Math.min(h(de) / h(I), 1)), J = be(() => h(de).toFixed(1));
  let we = be(() => s() == null), pe = be(() => r() !== null && r() !== void 0 ? r() : h(P));
  function Ie() {
    gi(() => {
      w(de, (performance.now() - h(_)) / 1e3), g && Ie();
    });
  }
  let ee = be(() => {
    let A = null;
    s() != null ? A = s().map((G) => {
      if (G.index != null && G.length != null)
        return G.index / G.length;
      if (G.progress != null)
        return G.progress;
    }) : A = null;
    let T, L = "";
    return A ? (T = A[A.length - 1], T === 0 ? L = "0" : L = "150ms") : T = void 0, {
      progress_level: A,
      last_progress_level: T,
      progress_bar_transition: L
    };
  });
  function ue() {
    g || (w(P, w(N, null), !0), w(_, performance.now(), !0), g = !0, Ie());
  }
  function Te() {
    w(P, w(N, null), !0), g && (g = !1);
  }
  _e(() => {
    t.status === "pending" ? ue() : te(() => {
      Te();
    });
  }), _e(() => {
    b && n() && (t.status === "pending" || t.status === "complete") && zo(b, t.autoscroll);
  }), _e(() => {
    h(pe) != null && h(P) !== h(pe) && (w(I, (performance.now() - h(_)) / 1e3 + h(pe)), w(N, h(I).toFixed(1), !0), w(P, h(pe), !0));
  });
  function Se() {
    w(B, !1);
  }
  _e(() => {
    te(() => {
      Se();
    }), t.status === "error" && o() && w(B, !0);
  }), _e(() => {
    t.status === "complete" && y() === "output" && S() && d() != null && (w(ge, d().toFixed(1), !0), w(V, S() === "full" ? "from cache" : "used cache", !0), w(le, c() != null && c() > d() && c() > 0, !0), w(ne, h(le) ? c().toFixed(1) : null, !0), w(R, !0), w($, !1), ce && clearTimeout(ce), Ve && clearTimeout(Ve), ce = setTimeout(
      () => {
        w($, !0), Ve = setTimeout(
          () => {
            w(R, !1), w($, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var ze = el(), Be = se(ze);
  let Ke, ve;
  var Oe = K(Be);
  {
    var Le = (A) => {
      var T = Xo(), L = K(T), G = j(L), X = K(G);
      {
        let k = be(() => t.i18n ? t.i18n("common.clear") : "Clear");
        cn(X, {
          get Icon() {
            return dn;
          },
          get label() {
            return h(k);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => E(null)
        });
      }
      Y(() => ae(L, `${E() ?? ""} `)), C(A, T);
    };
    W(Oe, (A) => {
      E() && m() && A(Le);
    });
  }
  var er = j(Oe, 2);
  {
    var tr = (A) => {
      var T = Qo(), L = se(T);
      {
        var G = (F) => {
          var Z = qo();
          let Ae;
          Y(() => Ae = xe(Z, "", Ae, {
            transform: `translateX(${(h(dt) || 0) * 100 - 100}%)`
          })), C(F, Z);
        };
        W(L, (F) => {
          u() === "default" && h(we) && a() === "full" && F(G);
        });
      }
      var X = j(L, 2);
      let k;
      var Q = K(X);
      {
        var he = (F) => {
          var Z = it(), Ae = se(Z);
          an(Ae, 17, s, rn, (vt, He) => {
            var Ot = it(), nr = se(Ot);
            {
              var Lt = (Xe) => {
                var mt = Wo(), Nt = se(mt);
                {
                  var ir = (Ne) => {
                    var qe = Pe();
                    Y((gt, bt) => ae(qe, `${gt ?? ""}/${bt ?? ""}`), [
                      () => mr(h(He).index || 0),
                      () => mr(h(He).length)
                    ]), C(Ne, qe);
                  }, et = (Ne) => {
                    var qe = Pe();
                    Y((gt) => ae(qe, gt), [() => mr(h(He).index || 0)]), C(Ne, qe);
                  };
                  W(Nt, (Ne) => {
                    h(He).length != null ? Ne(ir) : Ne(et, -1);
                  });
                }
                var tt = j(Nt);
                Y(() => ae(tt, ` ${h(He).unit ?? ""} |  `)), C(Xe, mt);
              };
              W(nr, (Xe) => {
                h(He).index != null && Xe(Lt);
              });
            }
            C(vt, Ot);
          }), C(F, Z);
        }, ke = (F) => {
          var Z = Pe();
          Y(() => ae(Z, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), C(F, Z);
        }, $e = (F) => {
          var Z = Pe("processing |");
          C(F, Z);
        };
        W(Q, (F) => {
          s() ? F(he) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? F(ke, 1) : t.queue_position === 0 && F($e, 2);
        });
      }
      var It = j(Q, 2);
      {
        var Ue = (F) => {
          var Z = Pe();
          Y(() => ae(Z, `${h(J) ?? ""}${r() ? `/${h(N)}` : ""}s`)), C(F, Z);
        };
        W(It, (F) => {
          i() && F(Ue);
        });
      }
      var Bt = j(X, 2);
      {
        var rr = (F) => {
          var Z = Yo(), Ae = K(Z), vt = K(Ae);
          {
            var He = (Xe) => {
              var mt = it(), Nt = se(mt);
              an(Nt, 17, s, rn, (ir, et, tt) => {
                var Ne = it(), qe = se(Ne);
                {
                  var gt = (bt) => {
                    var qr = Zo(), Wr = se(qr);
                    {
                      var yi = (ie) => {
                        var Me = Pe(" /");
                        C(ie, Me);
                      };
                      W(Wr, (ie) => {
                        tt !== 0 && ie(yi);
                      });
                    }
                    var Zr = j(Wr, 2);
                    {
                      var xi = (ie) => {
                        var Me = Pe();
                        Y(() => ae(Me, h(et).desc)), C(ie, Me);
                      };
                      W(Zr, (ie) => {
                        h(et).desc != null && ie(xi);
                      });
                    }
                    var Yr = j(Zr, 2);
                    {
                      var Ei = (ie) => {
                        var Me = Pe("-");
                        C(ie, Me);
                      };
                      W(Yr, (ie) => {
                        h(et).desc != null && h(ee).progress_level && h(ee).progress_level[tt] != null && ie(Ei);
                      });
                    }
                    var wi = j(Yr, 2);
                    {
                      var Ti = (ie) => {
                        var Me = Pe();
                        Y((Si) => ae(Me, `${Si ?? ""}%`), [
                          () => (100 * (h(ee).progress_level[tt] || 0)).toFixed(1)
                        ]), C(ie, Me);
                      };
                      W(wi, (ie) => {
                        h(ee).progress_level != null && ie(Ti);
                      });
                    }
                    C(bt, qr);
                  };
                  W(qe, (bt) => {
                    (h(et).desc != null || h(ee).progress_level && h(ee).progress_level[tt] != null) && bt(gt);
                  });
                }
                C(ir, Ne);
              }), C(Xe, mt);
            };
            W(vt, (Xe) => {
              s() != null && Xe(He);
            });
          }
          var Ot = j(Ae, 2), nr = K(Ot);
          let Lt;
          Y(() => Lt = xe(nr, "", Lt, {
            width: `${h(ee).last_progress_level * 100}%`,
            transition: h(ee).progress_bar_transition
          })), C(F, Z);
        }, pt = (F) => {
          {
            let Z = be(() => u() === "default");
            Go(F, {
              get margin() {
                return h(Z);
              }
            });
          }
        };
        W(Bt, (F) => {
          h(ee).last_progress_level != null ? F(rr) : a() === "full" && F(pt, 1);
        });
      }
      var bi = j(Bt, 2);
      {
        var _i = (F) => {
          var Z = Jo(), Ae = se(Z), vt = K(Ae), He = j(Ae, 2);
          wr(He, t, "additional-loading-text", {}), Y(() => ae(vt, l())), C(F, Z);
        };
        W(bi, (F) => {
          i() || F(_i);
        });
      }
      Y(() => k = De(X, 1, "progress-text svelte-124hqw6", null, k, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), C(A, T);
    }, fe = (A) => {
      var T = Ko(), L = se(T), G = K(L);
      {
        let he = be(() => t.i18n("common.clear"));
        cn(G, {
          get Icon() {
            return dn;
          },
          get label() {
            return h(he);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var X = j(L, 2), k = K(X), Q = j(X, 2);
      wr(Q, t, "error", {}), Y((he) => ae(k, he), [() => t.i18n("common.error")]), C(A, T);
    };
    W(er, (A) => {
      t.status === "pending" ? A(tr) : t.status === "error" && A(fe, 1);
    });
  }
  Vr(Be, (A) => b = A, () => b);
  var x = j(Be, 2);
  {
    var O = (A) => {
      var T = $o();
      let L, G;
      var X = K(T), k = j(X);
      {
        var Q = (ke) => {
          var $e = Pe();
          Y(() => ae($e, `~${h(ne) ?? ""}s
			→ `)), C(ke, $e);
        };
        W(k, (ke) => {
          h(le) && ke(Q);
        });
      }
      var he = j(k);
      Y(() => {
        L = De(T, 1, "cache-indicator svelte-124hqw6", null, L, { "fade-out": h($) }), G = xe(T, "", G, { position: f() ? "absolute" : "static" }), ae(X, `⚡ ${h(V) ?? ""}: `), ae(he, `${h(ge) ?? ""}s`);
      }), C(A, T);
    };
    W(x, (A) => {
      h(R) && A(O);
    });
  }
  Y(() => {
    Ke = De(Be, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", Ke, {
      "no-click": E() && m(),
      hide: h(Ee),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || p() || a() === "minimal" || E(),
      generating: t.status === "generating" && a() === "full",
      border: v()
    }), ve = xe(Be, "", ve, {
      position: f() ? "absolute" : "static",
      padding: f() ? "0" : "var(--size-8) 0"
    });
  }), C(e, ze), Wt();
}
const rl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, nl = [
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
], il = [
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
], al = [
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
rl([
  Object.fromEntries(nl.map((e) => [e, ["*"]])),
  Object.fromEntries(il.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(al.map((e) => [e, ["math:*"]]))
]);
qt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var sl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), ol = /* @__PURE__ */ re('<!> <div class="region-annotator svelte-r41nsf"><div class="toolbar svelte-r41nsf" role="group" aria-label="Region annotation tool"><button type="button">浏览</button> <button type="button">套索选择</button> <button type="button" class="clear svelte-r41nsf">清除 Draft</button></div> <div class="legend svelte-r41nsf"><span class="svelte-r41nsf"><i class="draft svelte-r41nsf"></i>黄色：Draft</span> <span class="svelte-r41nsf"><i class="saved svelte-r41nsf"></i>绿色：Saved Region</span></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function ul(e, t) {
  Zt(t, !0);
  const r = /* @__PURE__ */ Ya(t, sl), n = new Uo(r), i = 2048, a = 4096, o = 3;
  let s, u = null, l = null, f = null, p = null, v = null, E = q(Tt({})), m = q("browse"), y = q("请先生成版图 binary mask"), S = q(!1), d = q(!1), c = q(Tt([])), b = null, g = null, _ = "", P = 0, I = "", B = null, N = !1, R = !1, $ = q(!1);
  function ge(x) {
    return JSON.parse(JSON.stringify(x || {}));
  }
  function ne(x) {
    return typeof x == "number" ? `${x}px` : x || "520px";
  }
  function V(x, O, A) {
    return Math.max(O, Math.min(A, x));
  }
  function le() {
    return h(E).server_view || {};
  }
  function ce() {
    return h(E).client_intent || {};
  }
  function Ve(x = ce()) {
    const O = le();
    return JSON.stringify([
      x.session_id || "",
      x.layout_id || "",
      x.source_mask_hash || "",
      Number(O.natural_width || 0),
      Number(O.natural_height || 0)
    ]);
  }
  function Ee() {
    return Math.max(1, Number(le().natural_width || u?.naturalWidth || 1));
  }
  function de() {
    return Math.max(1, Number(le().natural_height || u?.naturalHeight || 1));
  }
  function dt() {
    return Math.min(1, i / Math.max(Ee(), de()));
  }
  function J(x, O, A) {
    const T = (G) => {
      O === P && A(G);
    };
    if (!x) {
      T(null);
      return;
    }
    const L = new Image();
    L.onload = () => T(L), L.onerror = () => T(null), L.src = x;
  }
  function we(x, O) {
    x === P && (w($, N && R && u !== null && l !== null, !0), N && R && w(y, h($) ? O : "当前 Layout 图像加载失败，套索已禁用", !0), fe());
  }
  function pe() {
    if (!l) {
      f = null;
      return;
    }
    const x = Ee(), O = de(), A = document.createElement("canvas");
    A.width = x, A.height = O;
    const T = A.getContext("2d", { willReadFrequently: !0 });
    if (!T) return;
    T.imageSmoothingEnabled = !1, T.drawImage(l, 0, 0, x, O);
    const L = T.getImageData(0, 0, x, O), G = document.createElement("canvas");
    G.width = x, G.height = O;
    const X = G.getContext("2d");
    if (!X) return;
    const k = X.createImageData(x, O);
    for (let Q = 0; Q < L.data.length; Q += 4) {
      const he = Math.max(L.data[Q], L.data[Q + 1], L.data[Q + 2]);
      L.data[Q + 3] > 0 && he >= 128 && (k.data[Q] = 45, k.data[Q + 1] = 160, k.data[Q + 2] = 255, k.data[Q + 3] = 80);
    }
    X.putImageData(k, 0, 0), f = G;
  }
  function Ie(x) {
    w(E, ge(x), !0);
    const O = ce(), A = le(), T = ++P, L = Ve(O), G = L !== I;
    I = L;
    const X = A.status || "Region 标注器已加载";
    if (w(m, O.tool_mode === "lasso" ? "lasso" : "browse", !0), w(
      c,
      Array.isArray(O.lasso_polygon) ? O.lasso_polygon.filter((k) => Array.isArray(k) && k.length === 2).map((k) => ({ x: Number(k[0]), y: Number(k[1]) })) : [],
      !0
    ), b !== null && s?.hasPointerCapture(b) && s.releasePointerCapture(b), w(S, !1), w(d, !1), b = null, B = null, g = null, p = null, v = null, G || A.enabled !== !0 ? (u = null, l = null, f = null, N = !1, R = !1, w($, !1)) : (N = u !== null, R = l !== null, w($, N && R, !0)), A.enabled !== !0) {
      w(y, X, !0), fe();
      return;
    }
    w(y, h($) ? X : "正在加载当前 Layout 图像…", !0), fe(), J(A.source_image, T, (k) => {
      u = k, N = !0, we(T, X);
    }), J(A.source_mask_image, T, (k) => {
      l = k, R = !0, pe(), we(T, X);
    }), J(A.saved_region_overlay_image, T, (k) => {
      p = k, fe();
    }), J(A.draft_region_overlay_image, T, (k) => {
      v = k, fe();
    });
  }
  _e(() => {
    const x = JSON.stringify(n.props.value || null);
    x !== _ && (_ = x, Ie(n.props.value));
  });
  function ee(x = !1) {
    w(
      E,
      Object.assign(Object.assign({}, h(E)), {
        client_intent: Object.assign(Object.assign({}, ce()), {
          tool_mode: h(m),
          lasso_polygon: h(c).map((O) => [O.x, O.y])
        })
      }),
      !0
    ), n.props.value = h(E), _ = JSON.stringify(h(E)), x && n.dispatch("input");
  }
  function ue(x) {
    h(S) && Le("绘制已取消"), w(m, x, !0), w(y, x === "lasso" ? "按住鼠标左键拖动套索" : "浏览模式：Canvas 只读", !0), ee(!1), fe();
  }
  function Te(x) {
    const O = s.getBoundingClientRect();
    return {
      x: V((x.clientX - O.left) / Math.max(1, O.width) * Ee(), 0, Ee() - 1),
      y: V((x.clientY - O.top) / Math.max(1, O.height) * de(), 0, de() - 1)
    };
  }
  function Se() {
    w(
      E,
      Object.assign(Object.assign({}, h(E)), {
        server_view: Object.assign(Object.assign({}, le()), { draft_region_overlay_image: "" })
      }),
      !0
    ), v = null;
  }
  function ze(x) {
    if (!(x.button !== 0 || h(m) !== "lasso" || le().enabled !== !0)) {
      if (!h($)) {
        w(y, "当前 Layout 图像尚未加载完成，不能开始套索"), fe();
        return;
      }
      x.preventDefault(), Se(), w(c, [Te(x)], !0), w(S, !0), w(d, !1), b = x.pointerId, B = I, g = { x: x.clientX, y: x.clientY }, s.setPointerCapture(x.pointerId), w(y, "正在绘制 Draft"), ee(!1), fe();
    }
  }
  function Be(x) {
    if (!h(S) || x.pointerId !== b || !g) return;
    if (B !== I) {
      Le("Layout 已切换，Draft 已清除");
      return;
    }
    if (x.preventDefault(), Math.hypot(x.clientX - g.x, x.clientY - g.y) < o) return;
    const A = Te(x);
    h(c).length < a ? w(c, [...h(c), A], !0) : (w(c, [...h(c).slice(0, -1), A], !0), w(d, !0)), g = { x: x.clientX, y: x.clientY }, w(
      y,
      h(d) ? `已达到 ${a} 点上限` : `Draft 点数：${h(c).length}`,
      !0
    ), fe();
  }
  function Ke(x) {
    const O = Te(x), A = h(c)[h(c).length - 1];
    A && A.x === O.x && A.y === O.y || (h(c).length < a ? w(c, [...h(c), O], !0) : w(c, [...h(c).slice(0, -1), O], !0));
  }
  function ve(x) {
    b !== null && s.hasPointerCapture(b) && s.releasePointerCapture(b), b = null, B = null, g = null, w(S, !1), x.preventDefault();
  }
  function Oe(x) {
    if (!h(S) || x.pointerId !== b) return;
    if (B !== I) {
      Le("Layout 已切换，Draft 已清除");
      return;
    }
    Ke(x), ve(x);
    const O = new Set(h(c).map((A) => `${A.x.toFixed(4)},${A.y.toFixed(4)}`));
    if (h(c).length < 3 || O.size < 3) {
      Le("套索点不足，Draft 已清除");
      return;
    }
    w(y, "正在生成权威 Draft 交集预览…"), ee(!0), fe();
  }
  function Le(x = "Draft 已清除") {
    b !== null && s?.hasPointerCapture(b) && s.releasePointerCapture(b), b = null, B = null, g = null, w(S, !1), w(d, !1), w(c, [], !0), Se(), w(y, x, !0), ee(!1), fe();
  }
  function er(x) {
    x.pointerId === b && Le("指针操作已取消，Draft 已清除");
  }
  function tr() {
    h(S) && Le("窗口失焦，Draft 已清除");
  }
  function fe() {
    if (!s) return;
    const x = Ee(), O = de(), A = dt();
    s.width = Math.max(1, Math.round(x * A)), s.height = Math.max(1, Math.round(O * A));
    const T = s.getContext("2d");
    if (T && (T.setTransform(A, 0, 0, A, 0, 0), T.clearRect(0, 0, x, O), u ? (T.imageSmoothingEnabled = !0, T.drawImage(u, 0, 0, x, O)) : (T.fillStyle = "#f8fafc", T.fillRect(0, 0, x, O), T.fillStyle = "#64748b", T.font = "18px sans-serif", T.fillText(le().enabled === !0 ? "正在加载当前 Layout 图像…" : "请先生成版图 binary mask", 24, 42)), T.imageSmoothingEnabled = !1, f && T.drawImage(f, 0, 0, x, O), p && T.drawImage(p, 0, 0, x, O), v && T.drawImage(v, 0, 0, x, O), h(c).length > 0)) {
      T.save(), T.beginPath(), T.moveTo(h(c)[0].x, h(c)[0].y);
      for (let L = 1; L < h(c).length; L += 1) T.lineTo(h(c)[L].x, h(c)[L].y);
      !h(S) && h(c).length >= 3 && T.closePath(), T.setLineDash([8, 5]), T.lineWidth = Math.max(2, x / 700), T.strokeStyle = "rgba(255, 220, 0, 0.98)", T.stroke(), T.restore();
    }
  }
  Ce("blur", ma, tr);
  {
    let x = be(() => h(S) ? "focus" : "base");
    es(e, {
      get visible() {
        return n.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return h(x);
      },
      padding: !1,
      get elem_id() {
        return n.shared.elem_id;
      },
      get elem_classes() {
        return n.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return n.shared.container;
      },
      get scale() {
        return n.shared.scale;
      },
      get min_width() {
        return n.shared.min_width;
      },
      children: (O, A) => {
        var T = ol(), L = se(T);
        tl(L, Qa(
          {
            get autoscroll() {
              return n.shared.autoscroll;
            },
            get i18n() {
              return n.i18n;
            }
          },
          () => n.shared.loading_status,
          {
            on_clear_status: () => n.dispatch("clear_status", n.shared.loading_status)
          }
        ));
        var G = j(L, 2), X = K(G), k = K(X);
        let Q;
        var he = j(k, 2);
        let ke;
        var $e = j(he, 2), It = j(X, 4), Ue = K(It);
        Vr(Ue, (pt) => s = pt, () => s);
        var Bt = j(It, 2), rr = K(Bt);
        Y(
          (pt) => {
            xe(G, pt), Q = De(k, 1, "svelte-r41nsf", null, Q, { active: h(m) === "browse" }), ke = De(he, 1, "svelte-r41nsf", null, ke, { active: h(m) === "lasso" }), xe(Ue, `cursor:${h(m) === "lasso" ? "crosshair" : "default"}`), ae(rr, `${h(y) ?? ""} · 点数 ${h(c).length ?? ""}/4096`);
          },
          [() => `min-height:${ne(n.props.height)}`]
        ), Ce("click", k, () => ue("browse")), Ce("click", he, () => ue("lasso")), Ce("click", $e, () => Le()), Ce("pointerdown", Ue, ze), Ce("pointermove", Ue, Be), Ce("pointerup", Ue, Oe), Ce("pointercancel", Ue, er), C(O, T);
      },
      $$slots: { default: !0 }
    });
  }
  Wt();
}
export {
  ul as default
};
