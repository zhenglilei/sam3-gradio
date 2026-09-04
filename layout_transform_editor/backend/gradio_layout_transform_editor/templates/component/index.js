import { i as Hn, g as mi, o as ba, n as vt, u as he, s as _a, r as cn, m as xt, a as S, b as o, t as Mn, d as ya, q as bi, c as _i, e as gt, f as Pr, h as wr, j as xa, T as Ea, k as wa, l as Tr, p as Et, v as On, w as Tt, x as yi, y as xi, z as Ei, A as rr, E as Ir, B as Lt, C as wi, D as ze, F as Un, G as Ta, H as Ti, I as Pn, J as Sa, K as Fn, L as Aa, M as Ha, N as nt, O as Si, P as Qr, Q as Ma, R as Oa, S as Pa, U as Ai, V as Ia, W as Na, X as Hi, Y as In, Z as jn, _ as Vn, $ as Ba, a0 as Ca, a1 as La, a2 as Ra, a3 as ka, a4 as Ga, a5 as Da, a6 as Ua, a7 as Nn, a8 as Fa, a9 as Mi, aa as Nr, ab as ja, ac as Va, ad as za, ae as Xa, af as qa, ag as Wa, ah as Bn, ai as Za, aj as Ya, ak as Ve, al as hn, am as dn, an as Ja, ao as Qa, ap as er, aq as Ka, ar as $a, as as es, at as ts, au as rs, av as Oi, aw as Yt, ax as Q, ay as ns, az as zn, aA as is, aB as Oe, aC as Br, aD as Cr, aE as j, aF as _e, aG as ne, aH as as, aI as K, aJ as ye, aK as Re, aL as ss } from "./render-CPsC5Y6J.js";
function Pi(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const os = [];
function ls(e, t = !1, r = !1) {
  return yr(e, /* @__PURE__ */ new Map(), "", os, null, r);
}
function yr(e, t, r, n, i = null, a = !1) {
  if (typeof e == "object" && e !== null) {
    var l = t.get(e);
    if (l !== void 0) return l;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (Hn(e)) {
      var u = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, u), i !== null && t.set(i, u);
      for (var h = 0; h < e.length; h += 1) {
        var c = e[h];
        h in e && (u[h] = yr(c, t, r, n, null, a));
      }
      return u;
    }
    if (mi(e) === ba) {
      u = {}, t.set(e, u), i !== null && t.set(i, u);
      for (var d of Object.keys(e))
        u[d] = yr(
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
      return yr(
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
function Cn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), vt;
  const n = he(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Nt = [];
function us(e, t) {
  return {
    subscribe: nr(e, t).subscribe
  };
}
function nr(e, t = vt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(u) {
    if (_a(e, u) && (e = u, r)) {
      const h = !Nt.length;
      for (const c of n)
        c[1](), Nt.push(c, e);
      if (h) {
        for (let c = 0; c < Nt.length; c += 2)
          Nt[c][0](Nt[c + 1]);
        Nt.length = 0;
      }
    }
  }
  function a(u) {
    i(u(
      /** @type {T} */
      e
    ));
  }
  function l(u, h = vt) {
    const c = [u, h];
    return n.add(c), n.size === 1 && (r = t(i, a) || vt), u(
      /** @type {T} */
      e
    ), () => {
      n.delete(c), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: l };
}
function Ut(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return us(r, (l, u) => {
    let h = !1;
    const c = [];
    let d = 0, _ = vt;
    const g = () => {
      if (d)
        return;
      _();
      const O = t(n ? c[0] : c, l, u);
      a ? l(O) : _ = typeof O == "function" ? O : vt;
    }, A = i.map(
      (O, H) => Cn(
        O,
        (x) => {
          c[H] = x, d &= ~(1 << H), h && g();
        },
        () => {
          d |= 1 << H;
        }
      )
    );
    return h = !0, g(), function() {
      cn(A), _(), h = !1;
    };
  });
}
function fs(e) {
  let t;
  return Cn(e, (r) => t = r)(), t;
}
let mr = !1, pn = /* @__PURE__ */ Symbol("unmounted");
function Xn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: xt(void 0),
    unsubscribe: vt
  };
  if (n.store !== e && !(pn in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = vt;
    else {
      var i = !0;
      n.unsubscribe = Cn(e, (a) => {
        i ? n.source.v = a : S(n.source, a);
      }), i = !1;
    }
  return e && pn in r ? fs(e) : o(n.source);
}
function cs() {
  const e = {};
  function t() {
    Mn(() => {
      for (var r in e)
        e[r].unsubscribe();
      ya(e, pn, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function hs(e) {
  var t = mr;
  try {
    return mr = !1, [e(), mr];
  } finally {
    mr = t;
  }
}
function ds(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, bi(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const ps = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function vs(e) {
  return (
    /** @type {string} */
    ps?.createHTML(e) ?? e
  );
}
function Ii(e) {
  var t = _i("template");
  return t.innerHTML = vs(e.replaceAll("<!>", "<!---->")), t.content;
}
function Rt(e, t) {
  var r = (
    /** @type {Effect} */
    Pr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function le(e, t) {
  var r = (t & Ea) !== 0, n = (t & wa) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Ii(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    wr(i)));
    var l = (
      /** @type {TemplateNode} */
      n || xa ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var u = (
        /** @type {TemplateNode} */
        wr(l)
      ), h = (
        /** @type {TemplateNode} */
        l.lastChild
      );
      Rt(u, h);
    } else
      Rt(l, l);
    return l;
  };
}
// @__NO_SIDE_EFFECTS__
function gs(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var l = (
        /** @type {DocumentFragment} */
        Ii(i)
      ), u = (
        /** @type {Element} */
        wr(l)
      );
      a = /** @type {Element} */
      wr(u);
    }
    var h = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Rt(h, h), h;
  };
}
// @__NO_SIDE_EFFECTS__
function Ni(e, t) {
  return /* @__PURE__ */ gs(e, t, "svg");
}
function rt(e = "") {
  {
    var t = gt(e + "");
    return Rt(t, t), t;
  }
}
function Ct() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = gt();
  return e.append(t, r), Rt(t, r), e;
}
function D(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Lr {
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
        Tr(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Tr(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, l] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const u = this.#e.get(l);
        u && (Et(u.effect), this.#e.delete(l));
      }
      for (const [a, l] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const u = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var c = document.createDocumentFragment();
            xi(l, c), c.append(gt()), this.#e.set(a, { effect: l, fragment: c });
          } else
            Et(l);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), On(l, u, !1)) : u();
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
      r.includes(n) || (Et(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      yi
    ), i = Ei();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), l = gt();
        a.append(l), this.#e.set(t, {
          effect: Tt(() => r(l)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          Tt(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [u, h] of this.#r)
        u === t ? n.unskip_effect(h) : n.skip_effect(h);
      for (const [u, h] of this.#e)
        u === t ? n.unskip_effect(h.effect) : n.skip_effect(h.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function ms(e, t, ...r) {
  var n = new Lr(e);
  rr(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Ir);
}
function bs(e) {
  Lt === null && Pi(), wi && Lt.l !== null ? ys(Lt).m.push(e) : ze(() => {
    const t = he(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function _s(e) {
  Lt === null && Pi(), bs(() => () => he(e));
}
function ys(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function re(e, t, r = !1) {
  var n = new Lr(e), i = r ? Ir : 0;
  function a(l, u) {
    n.ensure(l, u);
  }
  rr(() => {
    var l = !1;
    t((u, h = 0) => {
      l = !0, a(h, u);
    }), l || a(-1, null);
  }, i);
}
function Sr(e, t) {
  return t;
}
function xs(e, t, r) {
  for (var n = [], i = t.length, a, l = t.length, u = 0; u < i; u++) {
    let _ = t[u];
    On(
      _,
      () => {
        if (a) {
          if (a.pending.delete(_), a.done.add(_), a.pending.size === 0) {
            var g = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            vn(e, Pn(a.done)), g.delete(a), g.size === 0 && (e.outrogroups = null);
          }
        } else
          l -= 1;
      },
      !1
    );
  }
  if (l === 0) {
    var h = n.length === 0 && r !== null;
    if (h) {
      var c = (
        /** @type {Element} */
        r
      ), d = (
        /** @type {Element} */
        c.parentNode
      );
      Oa(d), d.append(c), e.items.clear();
    }
    vn(e, t, !h);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function vn(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const l of e.pending.values())
      for (const u of l)
        n.add(
          /** @type {EachItem} */
          e.items.get(u).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var a = t[i];
    if (n?.has(a)) {
      a.f |= nt;
      const l = document.createDocumentFragment();
      xi(a, l);
    } else
      Et(t[i], r);
  }
}
var qn;
function Ar(e, t, r, n, i, a = null) {
  var l = e, u = /* @__PURE__ */ new Map(), h = (t & Ai) !== 0;
  if (h) {
    var c = (
      /** @type {Element} */
      e
    );
    l = c.appendChild(gt());
  }
  var d = null, _ = Ti(() => {
    var b = r();
    return (
      /** @type {V[]} */
      Hn(b) ? b : b == null ? [] : Pn(b)
    );
  }), g, A = /* @__PURE__ */ new Map(), O = !0;
  function H(b) {
    (v.effect.f & Si) === 0 && (v.pending.delete(b), v.fallback = d, Es(v, g, l, t, n), d !== null && (g.length === 0 ? (d.f & nt) === 0 ? Tr(d) : (d.f ^= nt, Kt(d, null, l)) : On(d, () => {
      d = null;
    })));
  }
  function x(b) {
    v.pending.delete(b);
  }
  var p = rr(() => {
    g = /** @type {V[]} */
    o(_);
    for (var b = g.length, E = /* @__PURE__ */ new Set(), T = (
      /** @type {Batch} */
      yi
    ), I = Ei(), N = 0; N < b; N += 1) {
      var k = g[N], G = n(k, N), L = O ? null : u.get(G);
      L ? (L.v && Un(L.v, k), L.i && Un(L.i, N), I && T.unskip_effect(L.e)) : (L = ws(
        u,
        O ? l : qn ??= gt(),
        k,
        G,
        N,
        i,
        t,
        r
      ), O || (L.e.f |= nt), u.set(G, L)), E.add(G);
    }
    if (b === 0 && a && !d && (O ? d = Tt(() => a(l)) : (d = Tt(() => a(qn ??= gt())), d.f |= nt)), b > E.size && Ta(), !O)
      if (A.set(T, E), I) {
        for (const [ae, we] of u)
          E.has(ae) || T.skip_effect(we.e);
        T.oncommit(H), T.ondiscard(x);
      } else
        H(T);
    o(_);
  }), v = { effect: p, items: u, pending: A, outrogroups: null, fallback: d };
  O = !1;
}
function Jt(e) {
  for (; e !== null && (e.f & Ma) === 0; )
    e = e.next;
  return e;
}
function Es(e, t, r, n, i) {
  var a = (n & Ia) !== 0, l = t.length, u = e.items, h = Jt(e.effect.first), c, d = null, _, g = [], A = [], O, H, x, p;
  if (a)
    for (p = 0; p < l; p += 1)
      O = t[p], H = i(O, p), x = /** @type {EachItem} */
      u.get(H).e, (x.f & nt) === 0 && (x.nodes?.a?.measure(), (_ ??= /* @__PURE__ */ new Set()).add(x));
  for (p = 0; p < l; p += 1) {
    if (O = t[p], H = i(O, p), x = /** @type {EachItem} */
    u.get(H).e, e.outrogroups !== null)
      for (const L of e.outrogroups)
        L.pending.delete(x), L.done.delete(x);
    if ((x.f & Qr) !== 0 && (Tr(x), a && (x.nodes?.a?.unfix(), (_ ??= /* @__PURE__ */ new Set()).delete(x))), (x.f & nt) !== 0)
      if (x.f ^= nt, x === h)
        Kt(x, null, r);
      else {
        var v = d ? d.next : h;
        x === e.effect.last && (e.effect.last = x.prev), x.prev && (x.prev.next = x.next), x.next && (x.next.prev = x.prev), dt(e, d, x), dt(e, x, v), Kt(x, v, r), d = x, g = [], A = [], h = Jt(d.next);
        continue;
      }
    if (x !== h) {
      if (c !== void 0 && c.has(x)) {
        if (g.length < A.length) {
          var b = A[0], E;
          d = b.prev;
          var T = g[0], I = g[g.length - 1];
          for (E = 0; E < g.length; E += 1)
            Kt(g[E], b, r);
          for (E = 0; E < A.length; E += 1)
            c.delete(A[E]);
          dt(e, T.prev, I.next), dt(e, d, T), dt(e, I, b), h = b, d = I, p -= 1, g = [], A = [];
        } else
          c.delete(x), Kt(x, h, r), dt(e, x.prev, x.next), dt(e, x, d === null ? e.effect.first : d.next), dt(e, d, x), d = x;
        continue;
      }
      for (g = [], A = []; h !== null && h !== x; )
        (c ??= /* @__PURE__ */ new Set()).add(h), A.push(h), h = Jt(h.next);
      if (h === null)
        continue;
    }
    (x.f & nt) === 0 && g.push(x), d = x, h = Jt(x.next);
  }
  if (e.outrogroups !== null) {
    for (const L of e.outrogroups)
      L.pending.size === 0 && (vn(e, Pn(L.done)), e.outrogroups?.delete(L));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (h !== null || c !== void 0) {
    var N = [];
    if (c !== void 0)
      for (x of c)
        (x.f & Qr) === 0 && N.push(x);
    for (; h !== null; )
      (h.f & Qr) === 0 && h !== e.fallback && N.push(h), h = Jt(h.next);
    var k = N.length;
    if (k > 0) {
      var G = (n & Ai) !== 0 && l === 0 ? r : null;
      if (a) {
        for (p = 0; p < k; p += 1)
          N[p].nodes?.a?.measure();
        for (p = 0; p < k; p += 1)
          N[p].nodes?.a?.fix();
      }
      xs(e, N, G);
    }
  }
  a && bi(() => {
    if (_ !== void 0)
      for (x of _)
        x.nodes?.a?.apply();
  });
}
function ws(e, t, r, n, i, a, l, u) {
  var h = (l & Aa) !== 0 ? (l & Ha) === 0 ? xt(r, !1, !1) : Fn(r) : null, c = (l & Sa) !== 0 ? Fn(i) : null;
  return {
    v: h,
    i: c,
    e: Tt(() => (a(t, h ?? r, c ?? i, u), () => {
      e.delete(n);
    }))
  };
}
function Kt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & nt) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var l = (
        /** @type {TemplateNode} */
        Pa(n)
      );
      if (a.before(n), n === i)
        return;
      n = l;
    }
}
function dt(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function gn(e, t, r, n, i) {
  var a = t.$$slots?.[r], l = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], l = !0), a === void 0 || a(e, l ? () => n : n);
}
function Ts(e, t, r) {
  var n = new Lr(e);
  rr(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Ir);
}
const Ss = () => performance.now(), Ze = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Ss(),
  tasks: /* @__PURE__ */ new Set()
};
function Bi() {
  const e = Ze.now();
  Ze.tasks.forEach((t) => {
    t.c(e) || (Ze.tasks.delete(t), t.f());
  }), Ze.tasks.size !== 0 && Ze.tick(Bi);
}
function As(e) {
  let t;
  return Ze.tasks.size === 0 && Ze.tick(Bi), {
    promise: new Promise((r) => {
      Ze.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Ze.tasks.delete(t);
    }
  };
}
function Hs(e, t, r, n, i, a) {
  var l = null, u = (
    /** @type {TemplateNode} */
    e
  ), h = new Lr(u, !1);
  rr(() => {
    const c = t() || null;
    var d = c === "svg" ? Na : void 0;
    if (c === null) {
      h.ensure(null, null);
      return;
    }
    return h.ensure(c, (_) => {
      if (c) {
        if (l = _i(c, d), Rt(l, l), n) {
          var g = null, A = l.appendChild(gt());
          n(l, A), g?.remove();
        }
        Pr.nodes.end = l, _.before(l);
      }
    }), () => {
    };
  }, Ir), Mn(() => {
  });
}
function Ms(e, t) {
  var r = void 0, n;
  Hi(() => {
    r !== (r = t()) && (n && (Et(n), n = null), r && (n = Tt(() => {
      In(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Ci(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = Ci(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Os() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = Ci(e)) && (n && (n += " "), n += t);
  return n;
}
function Ps(e) {
  return typeof e == "object" ? Os(e) : e ?? "";
}
const Wn = [...`\x20\t\n\r\f\u00A0\v\uFEFF`];
function Is(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, l = 0; (l = n.indexOf(i, l)) >= 0; ) {
          var u = l + a;
          (l === 0 || Wn.includes(n[l - 1])) && (u === n.length || Wn.includes(n[u])) ? n = (l === 0 ? "" : n.substring(0, l)) + n.substring(u + 1) : l = u;
        }
  }
  return n === "" ? null : n;
}
function Zn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function Kr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Ns(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, l = 0, u = !1, h = [];
      n && h.push(...Object.keys(n).map(Kr)), i && h.push(...Object.keys(i).map(Kr));
      var c = 0, d = -1;
      const H = e.length;
      for (var _ = 0; _ < H; _++) {
        var g = e[_];
        if (u ? g === "/" && e[_ - 1] === "*" && (u = !1) : a ? a === g && (a = !1) : g === "/" && e[_ + 1] === "*" ? u = !0 : g === '"' || g === "'" ? a = g : g === "(" ? l++ : g === ")" && l--, !u && a === !1 && l === 0) {
          if (g === ":" && d === -1)
            d = _;
          else if (g === ";" || _ === H - 1) {
            if (d !== -1) {
              var A = Kr(e.substring(c, d).trim());
              if (!h.includes(A)) {
                g !== ";" && _++;
                var O = e.substring(c, _).trim();
                r += " " + O + ";";
              }
            }
            c = _ + 1, d = -1;
          }
        }
      }
    }
    return n && (r += Zn(n)), i && (r += Zn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function mt(e, t, r, n, i, a) {
  var l = (
    /** @type {any} */
    e[jn]
  );
  if (l !== r || l === void 0) {
    var u = Is(r, n, a);
    u == null ? e.removeAttribute("class") : t ? e.className = u : e.setAttribute("class", u), e[jn] = r;
  } else if (a && i !== a)
    for (var h in a) {
      var c = !!a[h];
      (i == null || c !== !!i[h]) && e.classList.toggle(h, c);
    }
  return a;
}
function $r(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ye(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[Vn]
  );
  if (i !== t) {
    var a = Ns(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[Vn] = t;
  } else n && (Array.isArray(n) ? ($r(e, r?.[0], n[0]), $r(e, r?.[1], n[1], "important")) : $r(e, r, n));
  return n;
}
function Hr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Hn(t))
      return Ba();
    for (var n of e.options)
      n.selected = t.includes(Yn(n));
    return;
  }
  for (n of e.options) {
    var i = Yn(n);
    if (Ca(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Li(e) {
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
  }), Mn(() => {
    t.disconnect();
  });
}
function Yn(e) {
  return "__value" in e ? e.__value : e.value;
}
const $t = /* @__PURE__ */ Symbol("class"), Bt = /* @__PURE__ */ Symbol("style"), Ri = /* @__PURE__ */ Symbol("is custom element"), ki = /* @__PURE__ */ Symbol("is html"), Bs = Nn ? "input" : "INPUT", Cs = Nn ? "option" : "OPTION", Ls = Nn ? "select" : "SELECT";
function Rs(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function wt(e, t, r, n) {
  var i = Gi(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[La] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Di(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function ks(e, t, r, n, i = !1, a = !1) {
  var l = Gi(e), u = l[Ri], h = !l[ki], c = t || {}, d = e.nodeName === Cs;
  for (var _ in t)
    _ in r || (r[_] = null);
  r.class ? r.class = Ps(r.class) : r.class = null, r[Bt] && (r.style ??= null);
  var g = Di(e);
  if (e.nodeName === Bs && "type" in r && ("value" in r || "__value" in r)) {
    var A = r.type;
    (A !== c.type || A === void 0 && e.hasAttribute("type")) && (c.type = A, wt(e, "type", A));
  }
  for (const E in r) {
    let T = r[E];
    if (d && E === "value" && T == null) {
      e.value = e.__value = "", c[E] = T;
      continue;
    }
    if (E === "class") {
      var O = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      mt(e, O, T, n, t?.[$t], r[$t]), c[E] = T, c[$t] = r[$t];
      continue;
    }
    if (E === "style") {
      Ye(e, T, t?.[Bt], r[Bt]), c[E] = T, c[Bt] = r[Bt];
      continue;
    }
    var H = c[E];
    if (!(T === H && !(T === void 0 && e.hasAttribute(E)))) {
      c[E] = T;
      var x = E[0] + E[1];
      if (x !== "$$")
        if (x === "on") {
          const I = {}, N = "$$" + E;
          let k = E.slice(2);
          var p = Xa(k);
          if (Fa(k) && (k = k.slice(0, -7), I.capture = !0), !p && H) {
            if (T != null) continue;
            e.removeEventListener(k, c[N], I), c[N] = null;
          }
          if (p)
            Mi(k, e, T), Nr([k]);
          else if (T != null) {
            let G = function(L) {
              c[E].call(this, L);
            };
            c[N] = ja(k, e, G, I);
          }
        } else if (E === "style")
          wt(e, E, T);
        else if (E === "autofocus")
          ds(
            /** @type {HTMLElement} */
            e,
            !!T
          );
        else if (!u && (E === "__value" || E === "value" && T != null))
          e.value = e.__value = T;
        else if (E === "selected" && d)
          Rs(
            /** @type {HTMLOptionElement} */
            e,
            T
          );
        else {
          var v = E;
          h || (v = Va(v));
          var b = v === "defaultValue" || v === "defaultChecked";
          if (T == null && !u && !b)
            if (l[E] = null, v === "value" || v === "checked") {
              let I = (
                /** @type {HTMLInputElement} */
                e
              );
              const N = t === void 0;
              if (v === "value") {
                let k = I.defaultValue;
                I.removeAttribute(v), I.defaultValue = k, I.value = I.__value = N ? k : null;
              } else {
                let k = I.defaultChecked;
                I.removeAttribute(v), I.defaultChecked = k, I.checked = N ? k : !1;
              }
            } else
              e.removeAttribute(E);
          else b || g.includes(v) && (u || typeof T != "string") ? (e[v] = T, v in l && (l[v] = za)) : typeof T != "function" && wt(e, v, T);
        }
    }
  }
  return c;
}
function Gs(e, t, r = [], n = [], i = [], a, l = !1, u = !1) {
  Da(i, r, n, (h) => {
    var c = void 0, d = {}, _ = e.nodeName === Ls, g = !1;
    if (Hi(() => {
      var O = t(...h.map(o)), H = ks(
        e,
        c,
        O,
        a,
        l,
        u
      );
      g && _ && "value" in O && Hr(
        /** @type {HTMLSelectElement} */
        e,
        O.value
      );
      for (let p of Object.getOwnPropertySymbols(d))
        O[p] || Et(d[p]);
      for (let p of Object.getOwnPropertySymbols(O)) {
        var x = O[p];
        p.description === Ua && (!c || x !== c[p]) && (d[p] && Et(d[p]), d[p] = Tt(() => Ms(e, () => x))), H[p] = x;
      }
      c = H;
    }), _) {
      var A = (
        /** @type {HTMLSelectElement} */
        e
      );
      In(() => {
        Hr(
          A,
          /** @type {Record<string | symbol, any>} */
          c.value,
          !0
        ), Li(A);
      });
    }
    g = !0;
  });
}
function Gi(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Ra] ??= {
      [Ri]: e.nodeName.includes("-"),
      [ki]: e.namespaceURI === ka
    }
  );
}
var Jn = /* @__PURE__ */ new Map();
function Di(e) {
  var t = e.getAttribute("is") || e.nodeName, r = Jn.get(t);
  if (r) return r;
  Jn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = Ga(i);
    for (var l in n)
      n[l].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      l !== "innerHTML" && l !== "textContent" && l !== "innerText" && r.push(l);
    i = mi(i);
  }
  return r;
}
function en(e, t) {
  return e === t || e?.[Bn] === t;
}
function Ln(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Lt.r
  ), a = (
    /** @type {Effect} */
    Pr
  );
  return In(() => {
    var l, u;
    return qa(() => {
      l = u, u = [], he(() => {
        en(r(...u), e) || (t(e, ...u), l && en(r(...l), e) && t(null, ...l));
      });
    }), () => {
      let h = a;
      for (; h !== i && h.parent !== null && h.parent.f & Wa; )
        h = h.parent;
      const c = () => {
        u && en(r(...u), e) && t(null, ...u);
      }, d = h.teardown;
      h.teardown = () => {
        c(), d?.();
      };
    };
  }), e;
}
function Ds(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Lt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Ve(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const l = hn(() => {
      let u = !1;
      const h = t.s;
      for (const c in h)
        h[c] !== a[c] && (a[c] = h[c], u = !0);
      return u && i++, i;
    });
    n = () => o(l);
  }
  r.b.length && Za(() => {
    Qn(t, n), cn(r.b);
  }), ze(() => {
    const i = he(() => r.m.map(Ya));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && ze(() => {
    Qn(t, n), cn(r.a);
  });
}
function Qn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) o(r);
  t();
}
const Us = {
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
function Fs(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Us
  );
}
const js = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Yt(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Yt(i) && (i = i());
      const a = dn(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Yt(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = dn(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Bn || t === Oi) return !1;
    for (let r of e.props)
      if (Yt(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Yt(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function Vs(...e) {
  return new Proxy({ props: e }, js);
}
function B(e, t, r, n) {
  var i = !wi || (r & $a) !== 0, a = (r & Ka) !== 0, l = (r & ts) !== 0, u = (
    /** @type {V} */
    n
  ), h = !0, c = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), d = () => l && i ? (c ??= hn(
    /** @type {() => V} */
    n
  ), o(c)) : (h && (h = !1, u = l ? he(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), u);
  let _;
  if (a) {
    var g = Bn in e || Oi in e;
    _ = dn(e, t)?.set ?? (g && t in e ? (E) => e[t] = E : void 0);
  }
  var A, O = !1;
  a ? [A, O] = hs(() => (
    /** @type {V} */
    e[t]
  )) : A = /** @type {V} */
  e[t], A === void 0 && n !== void 0 && (A = d(), _ && (i && Ja(), _(A)));
  var H;
  if (i ? H = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E === void 0 ? d() : (h = !0, E);
  } : H = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E !== void 0 && (u = /** @type {V} */
    void 0), E === void 0 ? u : E;
  }, i && (r & Qa) === 0)
    return H;
  if (_) {
    var x = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(E, T) {
        return arguments.length > 0 ? ((!i || !T || x || O) && _(T ? H() : E), E) : H();
      })
    );
  }
  var p = !1, v = ((r & es) !== 0 ? hn : Ti)(() => (p = !1, H()));
  a && o(v);
  var b = (
    /** @type {Effect} */
    Pr
  );
  return (
    /** @type {() => V} */
    (function(E, T) {
      if (arguments.length > 0) {
        const I = T ? o(v) : i && a ? er(E) : E;
        return S(v, I), p = !0, u !== void 0 && (u = I), E;
      }
      return rs && p || (b.f & Si) !== 0 ? v.v : o(v);
    })
  );
}
const zs = [
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
], Kn = {
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
zs.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: Kn[t][r],
    secondary: Kn[t][n]
  }
}), {});
function Xs(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var tn, $n;
function qs() {
  if ($n) return tn;
  $n = 1;
  var e = function(v) {
    return t(v) && !r(v);
  };
  function t(p) {
    return !!p && typeof p == "object";
  }
  function r(p) {
    var v = Object.prototype.toString.call(p);
    return v === "[object RegExp]" || v === "[object Date]" || a(p);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(p) {
    return p.$$typeof === i;
  }
  function l(p) {
    return Array.isArray(p) ? [] : {};
  }
  function u(p, v) {
    return v.clone !== !1 && v.isMergeableObject(p) ? H(l(p), p, v) : p;
  }
  function h(p, v, b) {
    return p.concat(v).map(function(E) {
      return u(E, b);
    });
  }
  function c(p, v) {
    if (!v.customMerge)
      return H;
    var b = v.customMerge(p);
    return typeof b == "function" ? b : H;
  }
  function d(p) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(p).filter(function(v) {
      return Object.propertyIsEnumerable.call(p, v);
    }) : [];
  }
  function _(p) {
    return Object.keys(p).concat(d(p));
  }
  function g(p, v) {
    try {
      return v in p;
    } catch {
      return !1;
    }
  }
  function A(p, v) {
    return g(p, v) && !(Object.hasOwnProperty.call(p, v) && Object.propertyIsEnumerable.call(p, v));
  }
  function O(p, v, b) {
    var E = {};
    return b.isMergeableObject(p) && _(p).forEach(function(T) {
      E[T] = u(p[T], b);
    }), _(v).forEach(function(T) {
      A(p, T) || (g(p, T) && b.isMergeableObject(v[T]) ? E[T] = c(T, b)(p[T], v[T], b) : E[T] = u(v[T], b));
    }), E;
  }
  function H(p, v, b) {
    b = b || {}, b.arrayMerge = b.arrayMerge || h, b.isMergeableObject = b.isMergeableObject || e, b.cloneUnlessOtherwiseSpecified = u;
    var E = Array.isArray(v), T = Array.isArray(p), I = E === T;
    return I ? E ? b.arrayMerge(p, v, b) : O(p, v, b) : u(v, b);
  }
  H.all = function(v, b) {
    if (!Array.isArray(v))
      throw new Error("first argument should be an array");
    return v.reduce(function(E, T) {
      return H(E, T, b);
    }, {});
  };
  var x = H;
  return tn = x, tn;
}
var Ws = qs();
const Zs = /* @__PURE__ */ Xs(Ws);
var mn = function(e, t) {
  return mn = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, mn(e, t);
};
function Rr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  mn(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var X = function() {
  return X = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, X.apply(this, arguments);
};
function Ys(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function rn(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function nn(e, t) {
  var r = t && t.cache ? t.cache : ro, n = t && t.serializer ? t.serializer : eo, i = t && t.strategy ? t.strategy : Ks;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function Js(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Qs(e, t, r, n) {
  var i = Js(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function Ui(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function Fi(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function Ks(e, t) {
  var r = e.length === 1 ? Qs : Ui;
  return Fi(e, this, r, t.cache.create(), t.serializer);
}
function $s(e, t) {
  return Fi(e, this, Ui, t.cache.create(), t.serializer);
}
var eo = function() {
  return JSON.stringify(arguments);
}, to = (
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
), ro = {
  create: function() {
    return new to();
  }
}, an = {
  variadic: $s
}, U;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(U || (U = {}));
var ie;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(ie || (ie = {}));
var kt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(kt || (kt = {}));
function ei(e) {
  return e.type === ie.literal;
}
function no(e) {
  return e.type === ie.argument;
}
function ji(e) {
  return e.type === ie.number;
}
function Vi(e) {
  return e.type === ie.date;
}
function zi(e) {
  return e.type === ie.time;
}
function Xi(e) {
  return e.type === ie.select;
}
function qi(e) {
  return e.type === ie.plural;
}
function io(e) {
  return e.type === ie.pound;
}
function Wi(e) {
  return e.type === ie.tag;
}
function Zi(e) {
  return !!(e && typeof e == "object" && e.type === kt.number);
}
function bn(e) {
  return !!(e && typeof e == "object" && e.type === kt.dateTime);
}
var Yi = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, ao = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function so(e) {
  var t = {};
  return e.replace(ao, function(r) {
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
var oo = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function lo(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(oo).filter(function(g) {
    return g.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], l = a.split("/");
    if (l.length === 0)
      throw new Error("Invalid number skeleton");
    for (var u = l[0], h = l.slice(1), c = 0, d = h; c < d.length; c++) {
      var _ = d[c];
      if (_.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: u, options: h });
  }
  return r;
}
function uo(e) {
  return e.replace(/^(.*?)-/, "");
}
var ti = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, Ji = /^(@+)?(\+|#+)?[rs]?$/g, fo = /(\*)(0+)|(#+)(0+)|(0+)/g, Qi = /^(0+)$/;
function ri(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(Ji, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function Ki(e) {
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
function co(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !Qi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function ni(e) {
  var t = {}, r = Ki(e);
  return r || t;
}
function ho(e) {
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
        t.style = "unit", t.unit = uo(i.options[0]);
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
        t = X(X(X({}, t), { notation: "scientific" }), i.options.reduce(function(h, c) {
          return X(X({}, h), ni(c));
        }, {}));
        continue;
      case "engineering":
        t = X(X(X({}, t), { notation: "engineering" }), i.options.reduce(function(h, c) {
          return X(X({}, h), ni(c));
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
        i.options[0].replace(fo, function(h, c, d, _, g, A) {
          if (c)
            t.minimumIntegerDigits = d.length;
          else {
            if (_ && g)
              throw new Error("We currently do not support maximum integer digits");
            if (A)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (Qi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (ti.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(ti, function(h, c, d, _, g, A) {
        return d === "*" ? t.minimumFractionDigits = c.length : _ && _[0] === "#" ? t.maximumFractionDigits = _.length : g && A ? (t.minimumFractionDigits = g.length, t.maximumFractionDigits = g.length + A.length) : (t.minimumFractionDigits = c.length, t.maximumFractionDigits = c.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = X(X({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = X(X({}, t), ri(a)));
      continue;
    }
    if (Ji.test(i.stem)) {
      t = X(X({}, t), ri(i.stem));
      continue;
    }
    var l = Ki(i.stem);
    l && (t = X(X({}, t), l));
    var u = co(i.stem);
    u && (t = X(X({}, t), u));
  }
  return t;
}
var br = {
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
function po(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var l = 1 + (a & 1), u = a < 2 ? 1 : 3 + (a >> 1), h = "a", c = vo(t);
      for ((c == "H" || c == "k") && (u = 0); u-- > 0; )
        r += h;
      for (; l-- > 0; )
        r = c + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function vo(e) {
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
  var i = br[n || ""] || br[r || ""] || br["".concat(r, "-001")] || br["001"];
  return i[0];
}
var sn, go = new RegExp("^".concat(Yi.source, "*")), mo = new RegExp("".concat(Yi.source, "*$"));
function F(e, t) {
  return { start: e, end: t };
}
var bo = !!String.prototype.startsWith && "_a".startsWith("a", 1), _o = !!String.fromCodePoint, yo = !!Object.fromEntries, xo = !!String.prototype.codePointAt, Eo = !!String.prototype.trimStart, wo = !!String.prototype.trimEnd, To = !!Number.isSafeInteger, So = To ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, _n = !0;
try {
  var Ao = ea("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  _n = ((sn = Ao.exec("a")) === null || sn === void 0 ? void 0 : sn[0]) === "a";
} catch {
  _n = !1;
}
var ii = bo ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), yn = _o ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", i = t.length, a = 0, l; i > a; ) {
      if (l = t[a++], l > 1114111)
        throw RangeError(l + " is not a valid code point");
      n += l < 65536 ? String.fromCharCode(l) : String.fromCharCode(((l -= 65536) >> 10) + 55296, l % 1024 + 56320);
    }
    return n;
  }
), ai = (
  // native
  yo ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], l = a[0], u = a[1];
        r[l] = u;
      }
      return r;
    }
  )
), $i = xo ? (
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
), Ho = Eo ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(go, "");
  }
), Mo = wo ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(mo, "");
  }
);
function ea(e, t) {
  return new RegExp(e, t);
}
var xn;
if (_n) {
  var si = ea("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  xn = function(t, r) {
    var n;
    si.lastIndex = r;
    var i = si.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  xn = function(t, r) {
    for (var n = []; ; ) {
      var i = $i(t, r);
      if (i === void 0 || ta(i) || No(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return yn.apply(void 0, n);
  };
var Oo = (
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
          var l = this.parseArgument(t, n);
          if (l.err)
            return l;
          i.push(l.val);
        } else {
          if (a === 125 && t > 0)
            break;
          if (a === 35 && (r === "plural" || r === "selectordinal")) {
            var u = this.clonePosition();
            this.bump(), i.push({
              type: ie.pound,
              location: F(u, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(U.UNMATCHED_CLOSING_TAG, F(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && En(this.peek() || 0)) {
            var l = this.parseTag(t, r);
            if (l.err)
              return l;
            i.push(l.val);
          } else {
            var l = this.parseLiteral(t, r);
            if (l.err)
              return l;
            i.push(l.val);
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
            type: ie.literal,
            value: "<".concat(i, "/>"),
            location: F(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var l = a.val, u = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !En(this.char()))
            return this.error(U.INVALID_TAG, F(u, this.clonePosition()));
          var h = this.clonePosition(), c = this.parseTagName();
          return i !== c ? this.error(U.UNMATCHED_CLOSING_TAG, F(h, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: ie.tag,
              value: i,
              children: l,
              location: F(n, this.clonePosition())
            },
            err: null
          } : this.error(U.INVALID_TAG, F(u, this.clonePosition())));
        } else
          return this.error(U.UNCLOSED_TAG, F(n, this.clonePosition()));
      } else
        return this.error(U.INVALID_TAG, F(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Io(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), i = ""; ; ) {
        var a = this.tryParseQuote(r);
        if (a) {
          i += a;
          continue;
        }
        var l = this.tryParseUnquoted(t, r);
        if (l) {
          i += l;
          continue;
        }
        var u = this.tryParseLeftAngleBracket();
        if (u) {
          i += u;
          continue;
        }
        break;
      }
      var h = F(n, this.clonePosition());
      return {
        val: { type: ie.literal, value: i, location: h },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Po(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return yn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), yn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(U.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(U.EMPTY_ARGUMENT, F(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(U.MALFORMED_ARGUMENT, F(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(U.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: ie.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: F(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(U.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(U.MALFORMED_ARGUMENT, F(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = xn(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), l = F(t, a);
      return { value: n, location: l };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, l = this.clonePosition(), u = this.parseIdentifierIfPossible().value, h = this.clonePosition();
      switch (u) {
        case "":
          return this.error(U.EXPECT_ARGUMENT_TYPE, F(l, h));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var c = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var d = this.clonePosition(), _ = this.parseSimpleArgStyleIfPossible();
            if (_.err)
              return _;
            var g = Mo(_.val);
            if (g.length === 0)
              return this.error(U.EXPECT_ARGUMENT_STYLE, F(this.clonePosition(), this.clonePosition()));
            var A = F(d, this.clonePosition());
            c = { style: g, styleLocation: A };
          }
          var O = this.tryParseArgumentClose(i);
          if (O.err)
            return O;
          var H = F(i, this.clonePosition());
          if (c && ii(c?.style, "::", 0)) {
            var x = Ho(c.style.slice(2));
            if (u === "number") {
              var _ = this.parseNumberSkeletonFromString(x, c.styleLocation);
              return _.err ? _ : {
                val: { type: ie.number, value: n, location: H, style: _.val },
                err: null
              };
            } else {
              if (x.length === 0)
                return this.error(U.EXPECT_DATE_TIME_SKELETON, H);
              var p = x;
              this.locale && (p = po(x, this.locale));
              var g = {
                type: kt.dateTime,
                pattern: p,
                location: c.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? so(p) : {}
              }, v = u === "date" ? ie.date : ie.time;
              return {
                val: { type: v, value: n, location: H, style: g },
                err: null
              };
            }
          }
          return {
            val: {
              type: u === "number" ? ie.number : u === "date" ? ie.date : ie.time,
              value: n,
              location: H,
              style: (a = c?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var b = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(U.EXPECT_SELECT_ARGUMENT_OPTIONS, F(b, X({}, b)));
          this.bumpSpace();
          var E = this.parseIdentifierIfPossible(), T = 0;
          if (u !== "select" && E.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(U.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, F(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var _ = this.tryParseDecimalInteger(U.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, U.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (_.err)
              return _;
            this.bumpSpace(), E = this.parseIdentifierIfPossible(), T = _.val;
          }
          var I = this.tryParsePluralOrSelectOptions(t, u, r, E);
          if (I.err)
            return I;
          var O = this.tryParseArgumentClose(i);
          if (O.err)
            return O;
          var N = F(i, this.clonePosition());
          return u === "select" ? {
            val: {
              type: ie.select,
              value: n,
              options: ai(I.val),
              location: N
            },
            err: null
          } : {
            val: {
              type: ie.plural,
              value: n,
              options: ai(I.val),
              offset: T,
              pluralType: u === "plural" ? "cardinal" : "ordinal",
              location: N
            },
            err: null
          };
        }
        default:
          return this.error(U.INVALID_ARGUMENT_TYPE, F(l, h));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(U.EXPECT_ARGUMENT_CLOSING_BRACE, F(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(U.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, F(i, this.clonePosition()));
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
        n = lo(t);
      } catch {
        return this.error(U.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: kt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? ho(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, l = !1, u = [], h = /* @__PURE__ */ new Set(), c = i.value, d = i.location; ; ) {
        if (c.length === 0) {
          var _ = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var g = this.tryParseDecimalInteger(U.EXPECT_PLURAL_ARGUMENT_SELECTOR, U.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (g.err)
              return g;
            d = F(_, this.clonePosition()), c = this.message.slice(_.offset, this.offset());
          } else
            break;
        }
        if (h.has(c))
          return this.error(r === "select" ? U.DUPLICATE_SELECT_ARGUMENT_SELECTOR : U.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, d);
        c === "other" && (l = !0), this.bumpSpace();
        var A = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? U.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : U.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, F(this.clonePosition(), this.clonePosition()));
        var O = this.parseMessage(t + 1, r, n);
        if (O.err)
          return O;
        var H = this.tryParseArgumentClose(A);
        if (H.err)
          return H;
        u.push([
          c,
          {
            value: O.val,
            location: F(A, this.clonePosition())
          }
        ]), h.add(c), this.bumpSpace(), a = this.parseIdentifierIfPossible(), c = a.value, d = a.location;
      }
      return u.length === 0 ? this.error(r === "select" ? U.EXPECT_SELECT_ARGUMENT_SELECTOR : U.EXPECT_PLURAL_ARGUMENT_SELECTOR, F(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !l ? this.error(U.MISSING_OTHER_CLAUSE, F(this.clonePosition(), this.clonePosition())) : { val: u, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var a = !1, l = 0; !this.isEOF(); ) {
        var u = this.char();
        if (u >= 48 && u <= 57)
          a = !0, l = l * 10 + (u - 48), this.bump();
        else
          break;
      }
      var h = F(i, this.clonePosition());
      return a ? (l *= n, So(l) ? { val: l, err: null } : this.error(r, h)) : this.error(t, h);
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
      var r = $i(this.message, t);
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
      if (ii(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && ta(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function En(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Po(e) {
  return En(e) || e === 47;
}
function Io(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function ta(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function No(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function wn(e) {
  e.forEach(function(t) {
    if (delete t.location, Xi(t) || qi(t))
      for (var r in t.options)
        delete t.options[r].location, wn(t.options[r].value);
    else ji(t) && Zi(t.style) || (Vi(t) || zi(t)) && bn(t.style) ? delete t.style.location : Wi(t) && wn(t.children);
  });
}
function Bo(e, t) {
  t === void 0 && (t = {}), t = X({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Oo(e, t).parse();
  if (r.err) {
    var n = SyntaxError(U[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || wn(r.val), r.val;
}
var Gt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Gt || (Gt = {}));
var kr = (
  /** @class */
  (function(e) {
    Rr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), oi = (
  /** @class */
  (function(e) {
    Rr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), Gt.INVALID_VALUE, a) || this;
    }
    return t;
  })(kr)
), Co = (
  /** @class */
  (function(e) {
    Rr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Gt.INVALID_VALUE, i) || this;
    }
    return t;
  })(kr)
), Lo = (
  /** @class */
  (function(e) {
    Rr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Gt.MISSING_VALUE, n) || this;
    }
    return t;
  })(kr)
), Pe;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(Pe || (Pe = {}));
function Ro(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== Pe.literal || r.type !== Pe.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function ko(e) {
  return typeof e == "function";
}
function xr(e, t, r, n, i, a, l) {
  if (e.length === 1 && ei(e[0]))
    return [
      {
        type: Pe.literal,
        value: e[0].value
      }
    ];
  for (var u = [], h = 0, c = e; h < c.length; h++) {
    var d = c[h];
    if (ei(d)) {
      u.push({
        type: Pe.literal,
        value: d.value
      });
      continue;
    }
    if (io(d)) {
      typeof a == "number" && u.push({
        type: Pe.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var _ = d.value;
    if (!(i && _ in i))
      throw new Lo(_, l);
    var g = i[_];
    if (no(d)) {
      (!g || typeof g == "string" || typeof g == "number") && (g = typeof g == "string" || typeof g == "number" ? String(g) : ""), u.push({
        type: typeof g == "string" ? Pe.literal : Pe.object,
        value: g
      });
      continue;
    }
    if (Vi(d)) {
      var A = typeof d.style == "string" ? n.date[d.style] : bn(d.style) ? d.style.parsedOptions : void 0;
      u.push({
        type: Pe.literal,
        value: r.getDateTimeFormat(t, A).format(g)
      });
      continue;
    }
    if (zi(d)) {
      var A = typeof d.style == "string" ? n.time[d.style] : bn(d.style) ? d.style.parsedOptions : n.time.medium;
      u.push({
        type: Pe.literal,
        value: r.getDateTimeFormat(t, A).format(g)
      });
      continue;
    }
    if (ji(d)) {
      var A = typeof d.style == "string" ? n.number[d.style] : Zi(d.style) ? d.style.parsedOptions : void 0;
      A && A.scale && (g = g * (A.scale || 1)), u.push({
        type: Pe.literal,
        value: r.getNumberFormat(t, A).format(g)
      });
      continue;
    }
    if (Wi(d)) {
      var O = d.children, H = d.value, x = i[H];
      if (!ko(x))
        throw new Co(H, "function", l);
      var p = xr(O, t, r, n, i, a), v = x(p.map(function(T) {
        return T.value;
      }));
      Array.isArray(v) || (v = [v]), u.push.apply(u, v.map(function(T) {
        return {
          type: typeof T == "string" ? Pe.literal : Pe.object,
          value: T
        };
      }));
    }
    if (Xi(d)) {
      var b = d.options[g] || d.options.other;
      if (!b)
        throw new oi(d.value, g, Object.keys(d.options), l);
      u.push.apply(u, xr(b.value, t, r, n, i));
      continue;
    }
    if (qi(d)) {
      var b = d.options["=".concat(g)];
      if (!b) {
        if (!Intl.PluralRules)
          throw new kr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Gt.MISSING_INTL_API, l);
        var E = r.getPluralRules(t, { type: d.pluralType }).select(g - (d.offset || 0));
        b = d.options[E] || d.options.other;
      }
      if (!b)
        throw new oi(d.value, g, Object.keys(d.options), l);
      u.push.apply(u, xr(b.value, t, r, n, i, g - (d.offset || 0)));
      continue;
    }
  }
  return Ro(u);
}
function Go(e, t) {
  return t ? X(X(X({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = X(X({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Do(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Go(e[n], t[n]), r;
  }, X({}, e)) : e;
}
function on(e) {
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
function Uo(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: nn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, rn([void 0], r, !1)))();
    }, {
      cache: on(e.number),
      strategy: an.variadic
    }),
    getDateTimeFormat: nn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, rn([void 0], r, !1)))();
    }, {
      cache: on(e.dateTime),
      strategy: an.variadic
    }),
    getPluralRules: nn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, rn([void 0], r, !1)))();
    }, {
      cache: on(e.pluralRules),
      strategy: an.variadic
    })
  };
}
var Fo = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var a = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(h) {
        var c = a.formatToParts(h);
        if (c.length === 1)
          return c[0].value;
        var d = c.reduce(function(_, g) {
          return !_.length || g.type !== Pe.literal || typeof _[_.length - 1] != "string" ? _.push(g.value) : _[_.length - 1] += g.value, _;
        }, []);
        return d.length <= 1 ? d[0] || "" : d;
      }, this.formatToParts = function(h) {
        return xr(a.ast, a.locales, a.formatters, a.formats, h, void 0, a.message);
      }, this.resolvedOptions = function() {
        var h;
        return {
          locale: ((h = a.resolvedLocale) === null || h === void 0 ? void 0 : h.toString()) || Intl.NumberFormat.supportedLocalesOf(a.locales)[0]
        };
      }, this.getAst = function() {
        return a.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var l = i || {};
        l.formatters;
        var u = Ys(l, ["formatters"]);
        this.ast = e.__parse(t, X(X({}, u), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Do(e.formats, n), this.formatters = i && i.formatters || Uo(this.formatterCache);
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
    }, e.__parse = Bo, e.formats = {
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
function jo(e, t) {
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
const pt = {}, Vo = (e, t, r) => r && (t in pt || (pt[t] = {}), e in pt[t] || (pt[t][e] = r), r), ra = (e, t) => {
  if (t == null)
    return;
  if (t in pt && e in pt[t])
    return pt[t][e];
  const r = Gr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = Xo(i, e);
    if (a)
      return Vo(e, t, a);
  }
};
let Rn;
const ir = nr({});
function zo(e) {
  return Rn[e] || null;
}
function na(e) {
  return e in Rn;
}
function Xo(e, t) {
  if (!na(e))
    return null;
  const r = zo(e);
  return jo(r, t);
}
function qo(e) {
  if (e == null)
    return;
  const t = Gr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (na(n))
      return n;
  }
}
function Wo(e, ...t) {
  delete pt[e], ir.update((r) => (r[e] = Zs.all([r[e] || {}, ...t]), r));
}
Ut(
  [ir],
  ([e]) => Object.keys(e)
);
ir.subscribe((e) => Rn = e);
const Er = {};
function Zo(e, t) {
  Er[e].delete(t), Er[e].size === 0 && delete Er[e];
}
function ia(e) {
  return Er[e];
}
function Yo(e) {
  return Gr(e).map((t) => {
    const r = ia(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Tn(e) {
  return e == null ? !1 : Gr(e).some(
    (t) => {
      var r;
      return (r = ia(t)) == null ? void 0 : r.size;
    }
  );
}
function Jo(e, t) {
  return Promise.all(
    t.map((n) => (Zo(e, n), n().then((i) => i.default || i)))
  ).then((n) => Wo(e, ...n));
}
const Qt = {};
function aa(e) {
  if (!Tn(e))
    return e in Qt ? Qt[e] : Promise.resolve();
  const t = Yo(e);
  return Qt[e] = Promise.all(
    t.map(
      ([r, n]) => Jo(r, n)
    )
  ).then(() => {
    if (Tn(e))
      return aa(e);
    delete Qt[e];
  }), Qt[e];
}
const Qo = {
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
}, Ko = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: Qo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, $o = Ko;
function Dt() {
  return $o;
}
const ln = nr(!1);
var el = Object.defineProperty, tl = Object.defineProperties, rl = Object.getOwnPropertyDescriptors, li = Object.getOwnPropertySymbols, nl = Object.prototype.hasOwnProperty, il = Object.prototype.propertyIsEnumerable, ui = (e, t, r) => t in e ? el(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, al = (e, t) => {
  for (var r in t || (t = {}))
    nl.call(t, r) && ui(e, r, t[r]);
  if (li)
    for (var r of li(t))
      il.call(t, r) && ui(e, r, t[r]);
  return e;
}, sl = (e, t) => tl(e, rl(t));
let Sn;
const Mr = nr(null);
function fi(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Gr(e, t = Dt().fallbackLocale) {
  const r = fi(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...fi(t)])] : r;
}
function St() {
  return Sn ?? void 0;
}
Mr.subscribe((e) => {
  Sn = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const ol = (e) => {
  if (e && qo(e) && Tn(e)) {
    const { loadingDelay: t } = Dt();
    let r;
    return typeof window < "u" && St() != null && t ? r = window.setTimeout(
      () => ln.set(!0),
      t
    ) : ln.set(!0), aa(e).then(() => {
      Mr.set(e);
    }).finally(() => {
      clearTimeout(r), ln.set(!1);
    });
  }
  return Mr.set(e);
}, Ft = sl(al({}, Mr), {
  set: ol
}), Dr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var ll = Object.defineProperty, Or = Object.getOwnPropertySymbols, sa = Object.prototype.hasOwnProperty, oa = Object.prototype.propertyIsEnumerable, ci = (e, t, r) => t in e ? ll(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, kn = (e, t) => {
  for (var r in t || (t = {}))
    sa.call(t, r) && ci(e, r, t[r]);
  if (Or)
    for (var r of Or(t))
      oa.call(t, r) && ci(e, r, t[r]);
  return e;
}, jt = (e, t) => {
  var r = {};
  for (var n in e)
    sa.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Or)
    for (var n of Or(e))
      t.indexOf(n) < 0 && oa.call(e, n) && (r[n] = e[n]);
  return r;
};
const tr = (e, t) => {
  const { formats: r } = Dt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, ul = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = tr("number", n)), new Intl.NumberFormat(r, i);
  }
), fl = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = tr("date", n) : Object.keys(i).length === 0 && (i = tr("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), cl = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = tr("time", n) : Object.keys(i).length === 0 && (i = tr("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), hl = (e = {}) => {
  var t = e, {
    locale: r = St()
  } = t, n = jt(t, [
    "locale"
  ]);
  return ul(kn({ locale: r }, n));
}, dl = (e = {}) => {
  var t = e, {
    locale: r = St()
  } = t, n = jt(t, [
    "locale"
  ]);
  return fl(kn({ locale: r }, n));
}, pl = (e = {}) => {
  var t = e, {
    locale: r = St()
  } = t, n = jt(t, [
    "locale"
  ]);
  return cl(kn({ locale: r }, n));
}, vl = Dr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = St()) => new Fo(e, t, Dt().formats, {
    ignoreTag: Dt().ignoreTag
  })
), gl = (e, t = {}) => {
  var r, n, i, a;
  let l = t;
  typeof e == "object" && (l = e, e = l.id);
  const {
    values: u,
    locale: h = St(),
    default: c
  } = l;
  if (h == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let d = ra(e, h);
  if (!d)
    d = (a = (i = (n = (r = Dt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: h, id: e, defaultValue: c })) != null ? i : c) != null ? a : e;
  else if (typeof d != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof d}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), d;
  if (!u)
    return d;
  let _ = d;
  try {
    _ = vl(d, h).format(u);
  } catch (g) {
    g instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      g.message
    );
  }
  return _;
}, ml = (e, t) => pl(t).format(e), bl = (e, t) => dl(t).format(e), _l = (e, t) => hl(t).format(e), yl = (e, t = St()) => ra(e, t);
Ut([Ft, ir], () => gl);
Ut([Ft], () => ml);
Ut([Ft], () => bl);
Ut([Ft], () => _l);
Ut([Ft, ir], () => yl);
const xl = "__i18n__", El = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], wl = [
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
function Tl(e) {
  return typeof e == "string" && e.includes(xl);
}
class Sl {
  load_component;
  #t = Q(er({}));
  get shared() {
    return o(this.#t);
  }
  set shared(t) {
    S(this.#t, t, !0);
  }
  #r = Q(er({}));
  get props() {
    return o(this.#r);
  }
  set props(t) {
    S(this.#r, t, !0);
  }
  #e = Q((t) => t);
  get i18n() {
    return o(this.#e);
  }
  set i18n(t) {
    S(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = wl;
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
    for (const n of El)
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
    ), ze(() => {
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
    }), Object.keys(this.translatable_props).length > 0 && Ft.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [a, l] = n.split("."), u = this.i18n(i);
        a === "shared" ? this.shared[l] = u : this.props[l] = u;
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
    return ls(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = Tl(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    ze(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
ns();
var Al = /* @__PURE__ */ Ni('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), hi = /* @__PURE__ */ le("<!> <!>", 1), Hl = /* @__PURE__ */ le('<div class="placeholder svelte-1stq1b1"></div>');
function Ml(e, t) {
  Cr(t, !1);
  let r = B(t, "height", 8, void 0), n = B(t, "min_height", 8, void 0), i = B(t, "max_height", 8, void 0), a = B(t, "width", 8, void 0), l = B(t, "elem_id", 8, ""), u = B(t, "elem_classes", 24, () => []), h = B(t, "variant", 8, "solid"), c = B(t, "border_mode", 8, "base"), d = B(t, "padding", 8, !0), _ = B(t, "type", 8, "normal"), g = B(t, "test_id", 8, void 0), A = B(t, "explicit_call", 8, !1), O = B(t, "container", 8, !0), H = B(t, "visible", 8, !0), x = B(t, "allow_overflow", 8, !0), p = B(t, "overflow_behavior", 8, "auto"), v = B(t, "scale", 8, null), b = B(t, "min_width", 8, 0), E = B(t, "flex", 12, !1), T = B(t, "resizable", 8, !1), I = B(t, "rtl", 8, !1), N = B(t, "fullscreen", 12, !1), k = B(t, "label", 8, void 0), G = xt(N()), L = xt(), ae = _() === "fieldset" ? "fieldset" : "div", we = xt(0), ue = xt(0), J = xt(null);
  function Ce(pe) {
    N() && pe.key === "Escape" && N(!1);
  }
  const Ie = (pe) => {
    if (pe !== void 0) {
      if (typeof pe == "number")
        return pe + "px";
      if (typeof pe == "string")
        return pe;
    }
  }, ut = (pe) => {
    let xe = pe.clientY;
    const Ne = ($) => {
      const ve = $.clientY - xe;
      xe = $.clientY, as(L, o(L).style.height = `${o(L).offsetHeight + ve}px`);
    }, q = () => {
      window.removeEventListener("mousemove", Ne), window.removeEventListener("mouseup", q);
    };
    window.addEventListener("mousemove", Ne), window.addEventListener("mouseup", q);
  };
  zn(
    () => (Ve(N()), o(G), o(L)),
    () => {
      N() !== o(G) && (S(G, N()), N() ? (S(J, o(L).getBoundingClientRect()), S(we, o(L).offsetHeight), S(ue, o(L).offsetWidth), window.addEventListener("keydown", Ce)) : (S(J, null), window.removeEventListener("keydown", Ce)));
    }
  ), zn(() => Ve(H()), () => {
    H() || E(!1);
  }), is(), Ds();
  var At = Ct(), de = Oe(At);
  {
    var ke = (pe) => {
      var xe = hi(), Ne = Oe(xe);
      Hs(Ne, () => ae, !1, (ve, V) => {
        Ln(ve, (Ge) => S(L, Ge), () => o(L)), Gs(
          ve,
          (Ge, it) => ({
            "data-testid": g(),
            id: l(),
            class: `block ${Ge ?? ""}`,
            dir: I() ? "rtl" : "ltr",
            "aria-label": k(),
            style: "",
            [$t]: {
              hidden: H() === "hidden",
              padded: d(),
              flex: E(),
              border_focus: c() === "focus",
              border_contrast: c() === "contrast",
              "hide-container": !A() && !O(),
              fullscreen: N(),
              animating: N() && o(J) !== null,
              "auto-margin": v() === null
            },
            [Bt]: it
          }),
          [
            () => (Ve(u()), he(() => u()?.join(" ") || "")),
            () => ({
              height: (Ve(N()), Ve(r()), he(() => N() ? void 0 : Ie(r()))),
              "min-height": (Ve(N()), Ve(n()), he(() => N() ? void 0 : Ie(n()))),
              "max-height": (Ve(N()), Ve(i()), he(() => N() ? void 0 : Ie(i()))),
              "--start-top": (o(J), he(() => o(J) ? `${o(J).top}px` : "0px")),
              "--start-left": (o(J), he(() => o(J) ? `${o(J).left}px` : "0px")),
              "--start-width": (o(J), he(() => o(J) ? `${o(J).width}px` : "0px")),
              "--start-height": (o(J), he(() => o(J) ? `${o(J).height}px` : "0px")),
              width: (Ve(N()), Ve(a()), he(() => N() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : Ie(a()))),
              "border-style": h(),
              overflow: x() ? p() : "hidden",
              "flex-grow": v(),
              "min-width": `calc(min(${b()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Je = hi(), fe = Oe(Je);
        gn(fe, t, "default", {});
        var se = j(fe, 2);
        {
          var Te = (Ge) => {
            var it = Al();
            _e("mousedown", it, ut), D(Ge, it);
          };
          re(se, (Ge) => {
            T() && Ge(Te);
          });
        }
        D(V, Je);
      });
      var q = j(Ne, 2);
      {
        var $ = (ve) => {
          var V = Hl();
          let Je;
          ne(() => Je = Ye(V, "", Je, {
            height: o(we) + "px",
            width: o(ue) + "px"
          })), D(ve, V);
        };
        re(q, (ve) => {
          N() && ve($);
        });
      }
      D(pe, xe);
    };
    re(de, (pe) => {
      (H() === !0 || H() === "hidden") && pe(ke);
    });
  }
  D(e, At), Br();
}
var Ol = /* @__PURE__ */ le('<span class="svelte-vvirtv"> </span>'), Pl = /* @__PURE__ */ le("<button><!> <div><!> <!></div></button>");
function di(e, t) {
  let r = B(t, "label", 3, ""), n = B(t, "show_label", 3, !1), i = B(t, "pending", 3, !1), a = B(t, "size", 3, "small"), l = B(t, "padded", 3, !0), u = B(t, "highlight", 3, !1), h = B(t, "disabled", 3, !1), c = B(t, "hasPopup", 3, !1), d = B(t, "color", 3, "var(--block-label-text-color)"), _ = B(t, "transparent", 3, !1), g = B(t, "background", 3, "var(--block-background-fill)"), A = B(t, "border", 3, "transparent"), O = Re(() => u() ? "var(--color-accent)" : d());
  var H = Pl();
  let x, p;
  var v = K(H);
  {
    var b = (G) => {
      var L = Ol(), ae = K(L);
      ne(() => ye(ae, r())), D(G, L);
    };
    re(v, (G) => {
      n() && G(b);
    });
  }
  var E = j(v, 2);
  let T;
  var I = K(E);
  Ts(I, () => t.Icon, (G, L) => {
    L(G, {});
  });
  var N = j(I, 2);
  {
    var k = (G) => {
      var L = Ct(), ae = Oe(L);
      ms(ae, () => t.children), D(G, L);
    };
    re(N, (G) => {
      t.children && G(k);
    });
  }
  ne(() => {
    x = mt(H, 1, "icon-button svelte-vvirtv", null, x, {
      pending: i(),
      padded: l(),
      highlight: u(),
      transparent: _()
    }), H.disabled = h(), wt(H, "aria-label", r()), wt(H, "aria-haspopup", c()), wt(H, "title", r()), p = Ye(H, "", p, {
      "--border-color": A(),
      color: !h() && o(O) ? o(O) : "var(--block-label-text-color)",
      "--bg-color": h() ? "auto" : g()
    }), T = mt(E, 1, "svelte-vvirtv", null, T, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Mi("click", H, function(...G) {
    t.onclick?.apply(this, G);
  }), D(e, H);
}
Nr(["click"]);
var Il = /* @__PURE__ */ Ni('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function pi(e) {
  var t = Il();
  D(e, t);
}
Nr(["click"]);
function un(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function vi(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function An(e, t, r, n) {
  if (typeof r == "number" || vi(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), l = e.opts.stiffness * i, u = e.opts.damping * a, h = (l - u) * e.inv_mass, c = (a + h) * e.dt;
    return Math.abs(c) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, vi(r) ? new Date(r.getTime() + c) : r + c);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          An(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = An(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function gi(e, t = {}) {
  const r = nr(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let l, u, h, c = (
    /** @type {T} */
    e
  ), d = (
    /** @type {T | undefined} */
    e
  ), _ = 1, g = 0, A = !1;
  function O(x, p = {}) {
    d = x;
    const v = h = {};
    return e == null || p.hard || H.stiffness >= 1 && H.damping >= 1 ? (A = !0, l = Ze.now(), c = x, r.set(e = d), Promise.resolve()) : (p.soft && (g = 1 / ((p.soft === !0 ? 0.5 : +p.soft) * 60), _ = 0), u || (l = Ze.now(), A = !1, u = As((b) => {
      if (A)
        return A = !1, u = null, !1;
      _ = Math.min(_ + g, 1);
      const E = Math.min(b - l, 1e3 / 30), T = {
        inv_mass: _,
        opts: H,
        settled: !0,
        dt: E * 60 / 1e3
      }, I = An(T, c, e, d);
      return l = b, c = /** @type {T} */
      e, r.set(e = /** @type {T} */
      I), T.settled && (u = null), !T.settled;
    })), new Promise((b) => {
      u.promise.then(() => {
        v === h && b();
      });
    }));
  }
  const H = {
    set: O,
    update: (x, p) => O(x(
      /** @type {T} */
      d,
      /** @type {T} */
      e
    ), p),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return H;
}
var Nl = /* @__PURE__ */ le('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function Bl(e, t) {
  Cr(t, !0);
  const r = () => Xn(h, "$top", i), n = () => Xn(c, "$bottom", i), [i, a] = cs();
  var l = this && this.__awaiter || function(b, E, T, I) {
    function N(k) {
      return k instanceof T ? k : new T(function(G) {
        G(k);
      });
    }
    return new (T || (T = Promise))(function(k, G) {
      function L(ue) {
        try {
          we(I.next(ue));
        } catch (J) {
          G(J);
        }
      }
      function ae(ue) {
        try {
          we(I.throw(ue));
        } catch (J) {
          G(J);
        }
      }
      function we(ue) {
        ue.done ? k(ue.value) : N(ue.value).then(L, ae);
      }
      we((I = I.apply(b, E || [])).next());
    });
  };
  let u = B(t, "margin", 3, !0);
  const h = gi([0, 0]), c = gi([0, 0]);
  let d = Q(!1);
  function _() {
    return l(this, void 0, void 0, function* () {
      yield Promise.all([h.set([125, 140]), c.set([-125, -140])]), yield Promise.all([h.set([-125, 140]), c.set([125, -140])]), yield Promise.all([h.set([-125, 0]), c.set([125, -0])]), yield Promise.all([h.set([125, 0]), c.set([-125, 0])]);
    });
  }
  function g() {
    return l(this, void 0, void 0, function* () {
      yield _(), o(d) || g();
    });
  }
  function A() {
    return l(this, void 0, void 0, function* () {
      yield Promise.all([h.set([125, 0]), c.set([-125, 0])]), g();
    });
  }
  ze(() => (A(), () => {
    S(d, !0);
  }));
  var O = Nl();
  let H;
  var x = K(O), p = K(x), v = j(p);
  ne(() => {
    H = mt(O, 1, "svelte-m6d381", null, H, { margin: u() }), Ye(p, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ye(v, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), D(e, O), Br(), a();
}
var Cl = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(l) {
      l(a);
    });
  }
  return new (r || (r = Promise))(function(a, l) {
    function u(d) {
      try {
        c(n.next(d));
      } catch (_) {
        l(_);
      }
    }
    function h(d) {
      try {
        c(n.throw(d));
      } catch (_) {
        l(_);
      }
    }
    function c(d) {
      d.done ? a(d.value) : i(d.value).then(u, h);
    }
    c((n = n.apply(e, t || [])).next());
  });
};
let _r = [], fn = !1;
const Ll = typeof window < "u", la = Ll ? window.requestAnimationFrame : (e) => {
};
function Rl(e) {
  return Cl(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (_r.push(t), !fn) fn = !0;
      else return;
      yield ss(), la(() => {
        let n = [0, 0];
        for (let i = 0; i < _r.length; i++) {
          const l = _r[i].getBoundingClientRect();
          (i === 0 || l.top + window.scrollY <= n[0]) && (n[0] = l.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), fn = !1, _r = [];
      });
    }
  });
}
var kl = /* @__PURE__ */ le('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), Gl = /* @__PURE__ */ le('<div class="eta-bar svelte-124hqw6"></div>'), Dl = /* @__PURE__ */ le("<!> ", 1), Ul = /* @__PURE__ */ le("<!> <!> <!> <!>", 1), Fl = /* @__PURE__ */ le('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), jl = /* @__PURE__ */ le('<p class="loading svelte-124hqw6"> </p> <!>', 1), Vl = /* @__PURE__ */ le("<!> <div><!> <!></div> <!> <!>", 1), zl = /* @__PURE__ */ le('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), Xl = /* @__PURE__ */ le("<div> <!> </div>"), ql = /* @__PURE__ */ le('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function Wl(e, t) {
  Cr(t, !0);
  let r = B(t, "eta", 3, null), n = B(t, "scroll_to_output", 3, !1), i = B(t, "timer", 3, !0), a = B(t, "show_progress", 3, "full"), l = B(t, "message", 3, null), u = B(t, "progress", 3, null), h = B(t, "variant", 3, "default"), c = B(t, "loading_text", 3, "Loading..."), d = B(t, "absolute", 3, !0), _ = B(t, "translucent", 3, !1), g = B(t, "border", 3, !1), A = B(t, "validation_error", 7, null), O = B(t, "show_validation_error", 3, !0), H = B(t, "type", 3, null), x = B(t, "used_cache", 3, null), p = B(t, "cache_duration", 3, null), v = B(t, "avg_time", 3, null), b, E = !1, T = Q(0), I = Q(null), N = Q(null), k = Q(!1), G = Q(null), L = Q(!1), ae = Q(!1), we = Q(null), ue = Q(null), J = Q("from cache"), Ce = Q(!1), Ie = null, ut = null;
  const At = Re(() => !(O() && A()) && (H() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let de = Q(0);
  const ke = Re(() => o(N) === null || o(N) <= 0 || !o(de) ? 0 : Math.min(o(de) / o(N), 1)), pe = Re(() => o(de).toFixed(1));
  let xe = Re(() => u() == null), Ne = Re(() => r() !== null && r() !== void 0 ? r() : o(I));
  function q() {
    la(() => {
      S(de, (performance.now() - o(T)) / 1e3), E && q();
    });
  }
  let $ = Re(() => {
    let Y = null;
    u() != null ? Y = u().map((me) => {
      if (me.index != null && me.length != null)
        return me.index / me.length;
      if (me.progress != null)
        return me.progress;
    }) : Y = null;
    let oe, ge = "";
    return Y ? (oe = Y[Y.length - 1], oe === 0 ? ge = "0" : ge = "150ms") : oe = void 0, {
      progress_level: Y,
      last_progress_level: oe,
      progress_bar_transition: ge
    };
  });
  function ve() {
    E || (S(I, S(G, null), !0), S(T, performance.now(), !0), E = !0, q());
  }
  function V() {
    S(I, S(G, null), !0), E && (E = !1);
  }
  ze(() => {
    t.status === "pending" ? ve() : he(() => {
      V();
    });
  }), ze(() => {
    b && n() && (t.status === "pending" || t.status === "complete") && Rl(b, t.autoscroll);
  }), ze(() => {
    o(Ne) != null && o(I) !== o(Ne) && (S(N, (performance.now() - o(T)) / 1e3 + o(Ne)), S(G, o(N).toFixed(1), !0), S(I, o(Ne), !0));
  });
  function Je() {
    S(k, !1);
  }
  ze(() => {
    he(() => {
      Je();
    }), t.status === "error" && l() && S(k, !0);
  }), ze(() => {
    t.status === "complete" && H() === "output" && x() && p() != null && (S(we, p().toFixed(1), !0), S(J, x() === "full" ? "from cache" : "used cache", !0), S(Ce, v() != null && v() > p() && v() > 0, !0), S(ue, o(Ce) ? v().toFixed(1) : null, !0), S(L, !0), S(ae, !1), Ie && clearTimeout(Ie), ut && clearTimeout(ut), Ie = setTimeout(
      () => {
        S(ae, !0), ut = setTimeout(
          () => {
            S(L, !1), S(ae, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var fe = ql(), se = Oe(fe);
  let Te, Ge;
  var it = K(se);
  {
    var Ur = (Y) => {
      var oe = kl(), ge = K(oe), me = j(ge), Se = K(me);
      {
        let Ae = Re(() => t.i18n ? t.i18n("common.clear") : "Clear");
        di(Se, {
          get Icon() {
            return pi;
          },
          get label() {
            return o(Ae);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => A(null)
        });
      }
      ne(() => ye(ge, `${A() ?? ""} `)), D(Y, oe);
    };
    re(it, (Y) => {
      A() && O() && Y(Ur);
    });
  }
  var ar = j(it, 2);
  {
    var Vt = (Y) => {
      var oe = Vl(), ge = Oe(oe);
      {
        var me = (W) => {
          var ee = Gl();
          let Ue;
          ne(() => Ue = Ye(ee, "", Ue, {
            transform: `translateX(${(o(ke) || 0) * 100 - 100}%)`
          })), D(W, ee);
        };
        re(ge, (W) => {
          h() === "default" && o(xe) && a() === "full" && W(me);
        });
      }
      var Se = j(ge, 2);
      let Ae;
      var Qe = K(Se);
      {
        var De = (W) => {
          var ee = Ct(), Ue = Oe(ee);
          Ar(Ue, 17, u, Sr, (ct, He) => {
            var Xe = Ct(), qt = Oe(Xe);
            {
              var Mt = (et) => {
                var ht = Dl(), Ot = Oe(ht);
                {
                  var Be = (Fe) => {
                    var at = rt();
                    ne((bt, _t) => ye(at, `${bt ?? ""}/${_t ?? ""}`), [
                      () => un(o(He).index || 0),
                      () => un(o(He).length)
                    ]), D(Fe, at);
                  }, ce = (Fe) => {
                    var at = rt();
                    ne((bt) => ye(at, bt), [() => un(o(He).index || 0)]), D(Fe, at);
                  };
                  re(Ot, (Fe) => {
                    o(He).length != null ? Fe(Be) : Fe(ce, -1);
                  });
                }
                var qe = j(Ot);
                ne(() => ye(qe, ` ${o(He).unit ?? ""} |  `)), D(et, ht);
              };
              re(qt, (et) => {
                o(He).index != null && et(Mt);
              });
            }
            D(ct, Xe);
          }), D(W, ee);
        }, ft = (W) => {
          var ee = rt();
          ne(() => ye(ee, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), D(W, ee);
        }, Ke = (W) => {
          var ee = rt("processing |");
          D(W, ee);
        };
        re(Qe, (W) => {
          u() ? W(De) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? W(ft, 1) : t.queue_position === 0 && W(Ke, 2);
        });
      }
      var $e = j(Qe, 2);
      {
        var or = (W) => {
          var ee = rt();
          ne(() => ye(ee, `${o(pe) ?? ""}${r() ? `/${o(G)}` : ""}s`)), D(W, ee);
        };
        re($e, (W) => {
          i() && W(or);
        });
      }
      var zt = j(Se, 2);
      {
        var Xt = (W) => {
          var ee = Fl(), Ue = K(ee), ct = K(Ue);
          {
            var He = (et) => {
              var ht = Ct(), Ot = Oe(ht);
              Ar(Ot, 17, u, Sr, (Be, ce, qe) => {
                var Fe = Ct(), at = Oe(Fe);
                {
                  var bt = (_t) => {
                    var fr = Ul(), cr = Oe(fr);
                    {
                      var Vr = (be) => {
                        var We = rt(" /");
                        D(be, We);
                      };
                      re(cr, (be) => {
                        qe !== 0 && be(Vr);
                      });
                    }
                    var hr = j(cr, 2);
                    {
                      var zr = (be) => {
                        var We = rt();
                        ne(() => ye(We, o(ce).desc)), D(be, We);
                      };
                      re(hr, (be) => {
                        o(ce).desc != null && be(zr);
                      });
                    }
                    var Wt = j(hr, 2);
                    {
                      var Xr = (be) => {
                        var We = rt("-");
                        D(be, We);
                      };
                      re(Wt, (be) => {
                        o(ce).desc != null && o($).progress_level && o($).progress_level[qe] != null && be(Xr);
                      });
                    }
                    var qr = j(Wt, 2);
                    {
                      var dr = (be) => {
                        var We = rt();
                        ne((Wr) => ye(We, `${Wr ?? ""}%`), [
                          () => (100 * (o($).progress_level[qe] || 0)).toFixed(1)
                        ]), D(be, We);
                      };
                      re(qr, (be) => {
                        o($).progress_level != null && be(dr);
                      });
                    }
                    D(_t, fr);
                  };
                  re(at, (_t) => {
                    (o(ce).desc != null || o($).progress_level && o($).progress_level[qe] != null) && _t(bt);
                  });
                }
                D(Be, Fe);
              }), D(et, ht);
            };
            re(ct, (et) => {
              u() != null && et(He);
            });
          }
          var Xe = j(Ue, 2), qt = K(Xe);
          let Mt;
          ne(() => Mt = Ye(qt, "", Mt, {
            width: `${o($).last_progress_level * 100}%`,
            transition: o($).progress_bar_transition
          })), D(W, ee);
        }, Ht = (W) => {
          {
            let ee = Re(() => h() === "default");
            Bl(W, {
              get margin() {
                return o(ee);
              }
            });
          }
        };
        re(zt, (W) => {
          o($).last_progress_level != null ? W(Xt) : a() === "full" && W(Ht, 1);
        });
      }
      var lr = j(zt, 2);
      {
        var ur = (W) => {
          var ee = jl(), Ue = Oe(ee), ct = K(Ue), He = j(Ue, 2);
          gn(He, t, "additional-loading-text", {}), ne(() => ye(ct, c())), D(W, ee);
        };
        re(lr, (W) => {
          i() || W(ur);
        });
      }
      ne(() => Ae = mt(Se, 1, "progress-text svelte-124hqw6", null, Ae, {
        "meta-text-center": h() === "center",
        "meta-text": h() === "default"
      })), D(Y, oe);
    }, sr = (Y) => {
      var oe = zl(), ge = Oe(oe), me = K(ge);
      {
        let De = Re(() => t.i18n("common.clear"));
        di(me, {
          get Icon() {
            return pi;
          },
          get label() {
            return o(De);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var Se = j(ge, 2), Ae = K(Se), Qe = j(Se, 2);
      gn(Qe, t, "error", {}), ne((De) => ye(Ae, De), [() => t.i18n("common.error")]), D(Y, oe);
    };
    re(ar, (Y) => {
      t.status === "pending" ? Y(Vt) : t.status === "error" && Y(sr, 1);
    });
  }
  Ln(se, (Y) => b = Y, () => b);
  var Fr = j(se, 2);
  {
    var jr = (Y) => {
      var oe = Xl();
      let ge, me;
      var Se = K(oe), Ae = j(Se);
      {
        var Qe = (ft) => {
          var Ke = rt();
          ne(() => ye(Ke, `~${o(ue) ?? ""}s
			→ `)), D(ft, Ke);
        };
        re(Ae, (ft) => {
          o(Ce) && ft(Qe);
        });
      }
      var De = j(Ae);
      ne(() => {
        ge = mt(oe, 1, "cache-indicator svelte-124hqw6", null, ge, { "fade-out": o(ae) }), me = Ye(oe, "", me, { position: d() ? "absolute" : "static" }), ye(Se, `⚡ ${o(J) ?? ""}: `), ye(De, `${o(we) ?? ""}s`);
      }), D(Y, oe);
    };
    re(Fr, (Y) => {
      o(L) && Y(jr);
    });
  }
  ne(() => {
    Te = mt(se, 1, `wrap ${h() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", Te, {
      "no-click": A() && O(),
      hide: o(At),
      translucent: h() === "center" && (t.status === "pending" || t.status === "error") || _() || a() === "minimal" || A(),
      generating: t.status === "generating" && a() === "full",
      border: g()
    }), Ge = Ye(se, "", Ge, {
      position: d() ? "absolute" : "static",
      padding: d() ? "0" : "var(--size-8) 0"
    });
  }), D(e, fe), Br();
}
const Zl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, Yl = [
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
], Jl = [
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
], Ql = [
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
Zl([
  Object.fromEntries(Yl.map((e) => [e, ["*"]])),
  Object.fromEntries(Jl.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(Ql.map((e) => [e, ["math:*"]]))
]);
Nr(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var Kl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), $l = /* @__PURE__ */ le("<option> </option>"), eu = /* @__PURE__ */ le('<button type="button"><span class="group-drag-handle svelte-r41nsf" title="拖动调整叠层顺序" aria-label="拖动调整叠层顺序">⠿</span> <span class="group-order svelte-r41nsf"></span> <span class="group-reorder-label svelte-r41nsf"> </span></button>'), tu = /* @__PURE__ */ le('<div class="group-reorder svelte-r41nsf" aria-label="调整 Label 叠层顺序"><div class="group-reorder-hint svelte-r41nsf">拖动调整叠层顺序</div> <div class="group-reorder-list svelte-r41nsf"></div></div>'), ru = /* @__PURE__ */ le('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select> <!></div>'), nu = /* @__PURE__ */ le('<!> <div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回视野</button></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function au(e, t) {
  Cr(t, !0);
  const r = /* @__PURE__ */ Fs(t, Kl), n = 2048, i = 35e-5, a = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], l = new Sl(r);
  let u, h = null, c = null, d = null, _ = null, g = /* @__PURE__ */ new Map(), A = /* @__PURE__ */ new Map(), O = Q(!1), H = Q(!1), x = Q("等待版图 mask"), p = Q(er({ enabled: !1 })), v = Q(er(ut())), b = Q(""), E = Q(!1), T = Q(!1), I = Q("crosshair"), N = "", k = 0, G = { x: 0, y: 0, center_x: 0, center_y: 0 }, L = { angle: 0, rotation: 0 }, ae = null, we = /* @__PURE__ */ new Set(), ue = Q(""), J = Q(""), Ce = Q("before"), Ie = Q(!1);
  function ut() {
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
  function At(s) {
    return JSON.parse(JSON.stringify(s || { enabled: !1 }));
  }
  function de(s) {
    const f = Object.assign(Object.assign({}, ut()), s || {});
    return Object.assign(Object.assign({}, f), {
      center_x: ke(f.center_x, 0),
      center_y: ke(f.center_y, 0),
      pivot_x: ke(f.pivot_x, 0),
      pivot_y: ke(f.pivot_y, 0),
      scale: xe(ke(f.scale, 1), 0.01, 20),
      rotation_deg: Ne(ke(f.rotation_deg, 0)),
      preview_alpha: xe(ke(f.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(ke(f.revision, 0)))
    });
  }
  function ke(s, f) {
    const m = Number(s);
    return Number.isFinite(m) ? m : f;
  }
  function pe(s) {
    return typeof s == "number" ? String(s) + "px" : s || "520px";
  }
  function xe(s, f, m) {
    return Math.max(f, Math.min(m, s));
  }
  function Ne(s) {
    let f = ((s + 180) % 360 + 360) % 360 - 180;
    return f === -180 && (f = 180), f;
  }
  function q(s) {
    return typeof s + ":" + String(s);
  }
  function $() {
    var s;
    return V() ? Array.isArray((s = o(p).group_view) === null || s === void 0 ? void 0 : s.groups) ? o(p).group_view.groups : [] : [];
  }
  function ve() {
    var s;
    return V() && ((s = o(p).group_view) === null || s === void 0 ? void 0 : s.allow_reorder) === !0;
  }
  function V() {
    return o(p).transform_mode === "label_groups" && !!o(p).group_view;
  }
  function Je() {
    for (const s of $())
      if (q(s.group_id) === o(b)) return s;
    return null;
  }
  function fe() {
    const s = Je();
    return s ? String(s.label || "Label " + String(s.group_id)) : "";
  }
  function se() {
    return Math.max(1, Number(o(p).target_width || h?.naturalWidth || 1));
  }
  function Te() {
    return Math.max(1, Number(o(p).target_height || h?.naturalHeight || 1));
  }
  function Ge() {
    var s;
    const f = (s = g.get(o(b))) === null || s === void 0 ? void 0 : s.image;
    return Math.max(1, Number(o(p).source_width || c?.naturalWidth || f?.naturalWidth || 1));
  }
  function it() {
    var s;
    const f = (s = g.get(o(b))) === null || s === void 0 ? void 0 : s.image;
    return Math.max(1, Number(o(p).source_height || c?.naturalHeight || f?.naturalHeight || 1));
  }
  function Ur() {
    return Math.min(1, n / Math.max(se(), Te()));
  }
  function ar(s = o(b)) {
    var f;
    if (V()) {
      const y = (f = g.get(s)) === null || f === void 0 ? void 0 : f.view.foreground_bbox_xyxy;
      if (Array.isArray(y) && y.length >= 4) return y.slice(0, 4).map(Number);
    }
    const m = o(p).foreground_bbox_xyxy;
    return Array.isArray(m) && m.length >= 4 ? m.slice(0, 4).map(Number) : [0, 0, Ge() - 1, it() - 1];
  }
  function Vt(s, f, m) {
    if (!s) {
      f === k && m(null);
      return;
    }
    const y = new Image();
    y.onload = () => {
      f === k && m(y);
    }, y.onerror = () => {
      f === k && m(null);
    }, y.src = s;
  }
  function sr(s, f) {
    const m = Math.max(1, Number(o(p).source_width || s.naturalWidth || 1)), y = Math.max(1, Number(o(p).source_height || s.naturalHeight || 1)), M = document.createElement("canvas");
    M.width = m, M.height = y;
    const w = M.getContext("2d", { willReadFrequently: !0 });
    if (!w) return null;
    w.imageSmoothingEnabled = !1, w.drawImage(s, 0, 0, m, y);
    const P = w.getImageData(0, 0, m, y), C = document.createElement("canvas");
    C.width = m, C.height = y;
    const R = C.getContext("2d");
    if (!R) return null;
    const Z = R.createImageData(m, y);
    for (let z = 0; z < P.data.length; z += 4) {
      const Ee = Math.max(P.data[z], P.data[z + 1], P.data[z + 2]);
      P.data[z + 3] > 0 && Ee >= 128 && (Z.data[z] = f[0], Z.data[z + 1] = f[1], Z.data[z + 2] = f[2], Z.data[z + 3] = 255);
    }
    return R.putImageData(Z, 0, 0), { mask: M, tint: C };
  }
  function Fr() {
    if (!c) {
      d = null, _ = null;
      return;
    }
    const s = sr(c, a[0]);
    d = s?.mask || null, _ = s?.tint || null;
  }
  function jr(s) {
    if (!s.image) {
      s.maskCanvas = null, s.tintCanvas = null, s.ready = !1;
      return;
    }
    const f = sr(s.image, s.color);
    s.maskCanvas = f?.mask || null, s.tintCanvas = f?.tint || null, s.ready = !!f;
  }
  function Y() {
    ae && (clearTimeout(ae), ae = null);
  }
  function oe(s, f) {
    if (!Array.isArray(f) || f.length !== s.length) return s;
    const m = /* @__PURE__ */ new Map();
    for (const w of s) {
      if (!w || w.group_id === void 0 || w.group_id === null) return s;
      const P = q(w.group_id);
      if (m.has(P)) return s;
      m.set(P, w);
    }
    const y = [], M = /* @__PURE__ */ new Set();
    for (const w of f) {
      if (typeof w != "string" && typeof w != "number") return s;
      const P = q(w), C = m.get(P);
      if (!C || M.has(P)) return s;
      M.add(P), y.push(C);
    }
    return y.length === s.length ? y : s;
  }
  function ge(s) {
    const f = o(p).group_view, m = Array.isArray(f?.groups) ? f.groups : [], y = o(p).group_intent, M = !!f && !!y && String(y.selection_signature || "") === String(f.selection_signature || ""), w = f?.allow_reorder === !0 && M ? oe(m, y?.ordered_group_ids) : m;
    w !== m && f && S(
      p,
      Object.assign(Object.assign({}, o(p)), {
        group_view: Object.assign(Object.assign({}, f), { groups: w })
      }),
      !0
    );
    const P = /* @__PURE__ */ new Map();
    if (M && Array.isArray(y?.transforms))
      for (const te of y.transforms)
        !te || te.group_id === void 0 || !te.transform || P.set(q(te.group_id), de(te.transform));
    const C = de(o(p).transform), R = /* @__PURE__ */ new Map(), Z = /* @__PURE__ */ new Map(), z = /* @__PURE__ */ new Set();
    for (let te = 0; te < w.length; te++) {
      const Le = w[te];
      if (!Le || Le.group_id === void 0 || Le.group_id === null) continue;
      const tt = q(Le.group_id);
      if (z.has(tt)) continue;
      z.add(tt);
      const Me = {
        view: Le,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: a[te % a.length]
      };
      R.set(tt, Me), Z.set(tt, de(P.get(tt) || C));
    }
    g = R, A = Z, we = /* @__PURE__ */ new Set();
    const Ee = M && y?.active_group_id !== void 0 && y.active_group_id !== null ? q(y.active_group_id) : "";
    S(
      b,
      R.has(Ee) ? Ee : R.keys().next().value || "",
      !0
    ), S(v, de(A.get(o(b)) || C), !0), o(b) && A.set(o(b), o(v)), c = null, d = null, _ = null, S(H, !1);
    for (const [te, Le] of R)
      Vt(Le.view.mask_image, s, (tt) => {
        const Me = g.get(te);
        !Me || Me !== Le || (Me.image = tt, jr(Me), Be());
      });
    S(
      x,
      o(p).status || (w.length > 0 ? "已加载 " + String(w.length) + " 个 Label；当前：" + fe() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function me(s) {
    Y(), k += 1;
    const f = k;
    S(p, At(s), !0), S(O, !1), g = /* @__PURE__ */ new Map(), A = /* @__PURE__ */ new Map(), S(b, ""), S(E, !1), S(T, !1), Vt(o(p).base_image, f, (m) => {
      h = m, S(O, !!m), Be();
    }), V() ? ge(f) : (S(v, de(o(p).transform), !0), S(x, o(p).status || "编辑器已加载", !0), S(H, !1), Vt(o(p).mask_image, f, (m) => {
      c = m, S(H, !!m), Fr(), Be();
    }));
  }
  ze(() => {
    const s = JSON.stringify(l.props.value || null);
    s !== N && (N = s, me(l.props.value));
  }), _s(() => {
    k += 1, Y();
  });
  function Se(s) {
    const f = Number(s.rotation_deg || 0) * Math.PI / 180, m = Number(s.scale || 1), y = Math.cos(f), M = Math.sin(f), w = m * y, P = m * M, C = Number(s.center_x || 0) - w * Number(s.pivot_x || 0) + P * Number(s.pivot_y || 0), R = Number(s.center_y || 0) - P * Number(s.pivot_x || 0) - w * Number(s.pivot_y || 0);
    return [w, P, -P, w, C, R];
  }
  function Ae(s, f, m = o(v)) {
    const [y, M, w, P, C, R] = Se(m);
    return { x: y * s + w * f + C, y: M * s + P * f + R };
  }
  function Qe(s, f, m = o(v)) {
    const [y, M, w, P, C, R] = Se(m), Z = y * P - M * w;
    if (Math.abs(Z) < 1e-9) return { x: -1, y: -1 };
    const z = s - C, Ee = f - R;
    return { x: (P * z - w * Ee) / Z, y: (-M * z + y * Ee) / Z };
  }
  function De(s) {
    const f = u.getBoundingClientRect();
    return {
      x: (s.clientX - f.left) / Math.max(1, f.width) * se(),
      y: (s.clientY - f.top) / Math.max(1, f.height) * Te()
    };
  }
  function ft(s, f, m) {
    if (!s) return !1;
    const y = Math.round(f), M = Math.round(m);
    if (y < 0 || M < 0 || y >= s.width || M >= s.height) return !1;
    const w = s.getContext("2d", { willReadFrequently: !0 });
    if (!w) return !1;
    const P = w.getImageData(y, M, 1, 1).data;
    return P[3] > 0 && Math.max(P[0], P[1], P[2]) >= 128;
  }
  function Ke(s) {
    return !V() || s === o(b) ? o(v) : A.get(s) || o(v);
  }
  function $e(s) {
    S(v, de(s), !0), V() && o(b) && A.set(o(b), o(v));
  }
  function or(s, f) {
    const m = Qe(s, f, o(v));
    return ft(d, m.x, m.y);
  }
  function zt(s, f, m) {
    const y = g.get(s);
    if (!y?.ready) return !1;
    const M = Qe(f, m, Ke(s));
    return ft(y.maskCanvas, M.x, M.y);
  }
  function Xt(s, f) {
    if (o(b) && zt(o(b), s, f)) return o(b);
    const m = $().map((y) => q(y.group_id)).reverse();
    for (const y of m)
      if (y !== o(b) && zt(y, s, f)) return y;
    return "";
  }
  function Ht(s = o(b), f = Ke(s)) {
    const [m, y, M, w] = ar(s);
    return [
      Ae(m, y, f),
      Ae(M, y, f),
      Ae(M, w, f),
      Ae(m, w, f)
    ];
  }
  function lr() {
    const s = Ht();
    let f = s[0];
    for (const w of s)
      (w.y < f.y || Math.abs(w.y - f.y) < 1e-6 && w.x > f.x) && (f = w);
    const m = Number(o(v).rotation_deg || 0) * Math.PI / 180, y = Math.cos(m - Math.PI / 4), M = Math.sin(m - Math.PI / 4);
    return { x: f.x + y * 34, y: f.y + M * 34 };
  }
  function ur() {
    const s = u?.getBoundingClientRect();
    return s ? Math.max(8, 14 * se() / Math.max(1, s.width)) : 14;
  }
  function W(s, f) {
    if (V() && !o(b)) return !1;
    const m = lr(), y = ur();
    return Math.hypot(s - m.x, f - m.y) <= y;
  }
  function ee(s) {
    const f = V() ? o(b) : "";
    $e(Object.assign(Object.assign({}, o(v)), {
      revision: Number(o(v).revision || 0) + 1,
      origin: s,
      scale: xe(Number(o(v).scale || 1), 0.01, 20),
      rotation_deg: Ne(Number(o(v).rotation_deg || 0))
    })), f && we.add(f);
  }
  function Ue() {
    const s = [];
    for (const f of $()) {
      const m = q(f.group_id);
      s.push({
        group_id: f.group_id,
        transform: Object.assign({}, de(Ke(m)))
      });
    }
    return s;
  }
  function ct() {
    const s = {
      enabled: o(p).enabled,
      transform: Object.assign({}, de(o(v))),
      target_width: o(p).target_width,
      target_height: o(p).target_height
    };
    V() && (s.transform_mode = "label_groups", s.group_intent = o(p).group_intent ? JSON.parse(JSON.stringify(o(p).group_intent)) : null), l.props.value = s, N = JSON.stringify(s);
  }
  function He(s, f, m, y = !0) {
    var M, w;
    if (!V()) return;
    m && ee(s);
    const P = o(p).group_view, C = Je(), R = $().filter((z) => we.has(q(z.group_id))).map((z) => z.group_id), Z = Number(((M = o(p).group_intent) === null || M === void 0 ? void 0 : M.transform_set_revision) || 0);
    S(
      p,
      Object.assign(Object.assign({}, o(p)), {
        group_intent: Object.assign(
          Object.assign(
            {
              selection_signature: String(P?.selection_signature || ""),
              transform_set_revision: Z + 1,
              active_group_id: (w = C?.group_id) !== null && w !== void 0 ? w : null,
              changed_group_ids: R
            },
            ve() ? {
              ordered_group_ids: $().map((z) => z.group_id)
            } : {}
          ),
          { transforms: Ue() }
        )
      }),
      !0
    ), S(x, f, !0), ct(), y && (Y(), l.dispatch("change")), Be();
  }
  function Xe(s, f, m = !0) {
    if (V()) {
      He(s, f, !0, m);
      return;
    }
    ee(s), S(
      p,
      Object.assign(Object.assign({}, o(p)), {
        enabled: !0,
        transform: Object.assign({}, o(v)),
        status: f
      }),
      !0
    ), S(x, f, !0), ct(), m && (Y(), l.dispatch("change")), Be();
  }
  function qt(s, f, m = 140) {
    Xe(s, f, !1), Y(), ae = setTimeout(
      () => {
        ae = null, l.dispatch("change");
      },
      m
    );
  }
  function Mt(s, f, m) {
    if (h && o(O)) {
      s.imageSmoothingEnabled = !0, s.drawImage(h, 0, 0, f, m);
      return;
    }
    s.fillStyle = "#f8fafc", s.fillRect(0, 0, f, m), s.fillStyle = "#64748b", s.font = "18px sans-serif", s.fillText("请先上传图像", 24, 42);
  }
  function et(s, f, m, y) {
    var M;
    s.save(), s.globalAlpha = xe(Number((M = y.preview_alpha) !== null && M !== void 0 ? M : 0.35), 0, 1);
    const [w, P, C, R, Z, z] = Se(y);
    s.setTransform(f * w, f * P, f * C, f * R, f * Z, f * z), s.imageSmoothingEnabled = !1, s.drawImage(m, 0, 0, m.width, m.height), s.restore();
  }
  function ht(s, f) {
    const m = Ht();
    s.save(), s.lineJoin = "round", s.lineWidth = Math.max(2.5, f / 700), s.strokeStyle = "rgba(0,0,0,0.82)", s.beginPath(), s.moveTo(m[0].x, m[0].y);
    for (let w = 1; w < m.length; w++) s.lineTo(m[w].x, m[w].y);
    s.closePath(), s.stroke(), s.lineWidth = Math.max(1.8, f / 1e3), s.strokeStyle = "#00ff66", s.stroke();
    const y = lr(), M = m.reduce((w, P) => P.y < w.y || Math.abs(P.y - w.y) < 1e-6 && P.x > w.x ? P : w, m[0]);
    s.strokeStyle = "#0f172a", s.lineWidth = Math.max(2, f / 900), s.beginPath(), s.moveTo(M.x, M.y), s.lineTo(y.x, y.y), s.stroke(), s.fillStyle = o(T) ? "#ffb000" : "#ffffff", s.strokeStyle = "#00ff66", s.lineWidth = Math.max(2, f / 900), s.beginPath(), s.arc(y.x, y.y, ur(), 0, Math.PI * 2), s.fill(), s.stroke(), s.restore();
  }
  function Ot(s, f, m) {
    const y = g.get(f);
    if (!y) return;
    const M = Ht(f, Ke(f)), w = Math.min(...M.map((Ee) => Ee.x)), P = Math.min(...M.map((Ee) => Ee.y)), C = String(y.view.label || "Label " + String(y.view.group_id));
    s.save(), s.font = "600 13px sans-serif";
    const R = s.measureText(C).width + 12, Z = xe(w, 2, Math.max(2, se() - R - 2)), z = xe(P - 23, 2, Math.max(2, Te() - 22));
    s.fillStyle = m ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.76)", s.fillRect(Z, z, R, 20), s.fillStyle = "#ffffff", s.fillText(C, Z + 6, z + 14), s.restore();
  }
  function Be() {
    var s, f;
    if (!u) return;
    const m = se(), y = Te(), M = Ur();
    u.width = Math.max(1, Math.round(m * M)), u.height = Math.max(1, Math.round(y * M));
    const w = u.getContext("2d");
    if (w && (w.setTransform(M, 0, 0, M, 0, 0), w.clearRect(0, 0, m, y), Mt(w, m, y), o(p).enabled !== !1)) {
      if (V()) {
        const C = $().map((R) => q(R.group_id)).filter((R) => R !== o(b));
        o(b) && C.push(o(b));
        for (const R of C) {
          const Z = g.get(R);
          Z?.ready && Z.tintCanvas && et(w, M, Z.tintCanvas, Ke(R));
        }
        w.setTransform(M, 0, 0, M, 0, 0);
        for (const R of C)
          !((s = g.get(R)) === null || s === void 0) && s.ready && Ot(w, R, R === o(b));
        o(b) && (!((f = g.get(o(b))) === null || f === void 0) && f.ready) && ht(w, m);
        return;
      }
      _ && o(H) && (et(w, M, _, o(v)), w.setTransform(M, 0, 0, M, 0, 0), ht(w, m));
    }
  }
  function ce() {
    var s;
    return o(p).enabled ? V() ? !!o(b) && !!(!((s = g.get(o(b))) === null || s === void 0) && s.ready) : o(H) && !!d : !1;
  }
  function qe(s) {
    return !V() || !g.has(s) || s === o(b) ? !1 : (o(b) && A.set(o(b), de(o(v))), S(b, s, !0), S(v, de(A.get(s)), !0), A.set(s, o(v)), S(x, "当前 Label：" + fe()), Be(), !0);
  }
  function Fe() {
    S(ue, ""), S(J, ""), S(Ce, "before");
  }
  function at(s, f) {
    ve() && (S(ue, f, !0), S(J, ""), S(Ce, "before"), S(Ie, !1), s.dataTransfer && (s.dataTransfer.effectAllowed = "move", s.dataTransfer.setData("text/plain", f)));
  }
  function bt(s, f) {
    if (!ve() || !o(ue)) return;
    s.preventDefault(), s.dataTransfer && (s.dataTransfer.dropEffect = "move");
    const y = s.currentTarget.getBoundingClientRect();
    S(J, f, !0), S(Ce, s.clientY < y.top + y.height / 2 ? "before" : "after", !0);
  }
  function _t(s, f) {
    var m;
    if (!ve()) return;
    s.preventDefault();
    const y = o(ue) || ((m = s.dataTransfer) === null || m === void 0 ? void 0 : m.getData("text/plain")) || "", M = o(J) === f ? o(Ce) : "before", w = cr(y, f, M);
    Fe(), w && (S(Ie, !0), He("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", !1, !0), setTimeout(
      () => {
        S(Ie, !1);
      },
      0
    ));
  }
  function fr() {
    Fe();
  }
  function cr(s, f, m) {
    if (!ve() || s === f) return !1;
    const y = o(p).group_view, M = $(), w = M.findIndex((te) => q(te.group_id) === s), P = M.findIndex((te) => q(te.group_id) === f);
    if (!y || w < 0 || P < 0) return !1;
    const C = M.slice(), [R] = C.splice(w, 1);
    if (!R) return !1;
    const Z = C.findIndex((te) => q(te.group_id) === f);
    if (Z < 0) return !1;
    const z = Z + (m === "after" ? 1 : 0);
    return C.splice(z, 0, R), C.some((te, Le) => q(te.group_id) !== q(M[Le].group_id)) ? (S(
      p,
      Object.assign(Object.assign({}, o(p)), {
        group_view: Object.assign(Object.assign({}, y), { groups: C })
      }),
      !0
    ), !0) : !1;
  }
  function Vr(s, f) {
    if (o(Ie)) {
      S(Ie, !1);
      return;
    }
    const m = s.target;
    m?.closest(".group-drag-handle") || qe(f) && He("select_group", "已选择 Label：" + fe(), !1, !0);
  }
  function hr(s) {
    const f = s.currentTarget.value;
    qe(f) && He("select_group", "已选择 Label：" + fe(), !1, !0);
  }
  function zr(s) {
    if (s.button !== 0) return;
    if (!ce()) {
      S(x, "请先启用并加载版图 mask"), Be();
      return;
    }
    Y();
    const f = De(s);
    if (W(f.x, f.y)) {
      S(I, "grabbing"), S(T, !0), L = {
        angle: Math.atan2(f.y - o(v).center_y, f.x - o(v).center_x) * 180 / Math.PI,
        rotation: Number(o(v).rotation_deg || 0)
      }, u.setPointerCapture(s.pointerId);
      return;
    }
    if (V()) {
      const m = Xt(f.x, f.y);
      if (!m) {
        S(x, "请点中任一 Label mask 前景后拖动"), Be();
        return;
      }
      qe(m);
    } else if (!or(f.x, f.y)) {
      S(x, "请点中版图 mask 前景后拖动"), Be();
      return;
    }
    S(I, "grabbing"), S(E, !0), G = {
      x: f.x,
      y: f.y,
      center_x: Number(o(v).center_x || 0),
      center_y: Number(o(v).center_y || 0)
    }, u.setPointerCapture(s.pointerId);
  }
  function Wt(s) {
    if (!ce()) {
      S(I, "not-allowed");
      return;
    }
    const f = V() ? !!Xt(s.x, s.y) : or(s.x, s.y);
    if (W(s.x, s.y) || f) {
      S(I, "grab");
      return;
    }
    S(I, "crosshair");
  }
  function Xr(s) {
    const f = De(s);
    if (!o(E) && !o(T)) {
      Wt(f);
      return;
    }
    if (S(I, "grabbing"), o(E))
      $e(Object.assign(Object.assign({}, o(v)), {
        center_x: G.center_x + f.x - G.x,
        center_y: G.center_y + f.y - G.y
      })), S(x, "正在拖动 " + (V() ? fe() : "版图") + "；松开后同步变换");
    else if (o(T)) {
      const m = Math.atan2(f.y - o(v).center_y, f.x - o(v).center_x) * 180 / Math.PI;
      $e(Object.assign(Object.assign({}, o(v)), {
        rotation_deg: Ne(L.rotation + m - L.angle)
      })), S(x, "正在旋转 " + (V() ? fe() : "版图") + "；松开后同步变换");
    }
    Be();
  }
  function qr() {
    !o(E) && !o(T) && S(I, "crosshair");
  }
  function dr(s) {
    if (o(E) || o(T)) {
      S(E, !1), S(T, !1);
      try {
        u.releasePointerCapture(s.pointerId);
      } catch {
      }
      Wt(De(s));
      const f = V() ? fe() : "Canvas";
      Xe("canvas", f + " 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function be(s) {
    if (!ce()) return;
    s.preventDefault();
    const f = De(s);
    if (V()) {
      const C = Xt(f.x, f.y);
      C && qe(C);
    }
    const m = Qe(f.x, f.y, o(v)), y = Math.exp(-s.deltaY * i), M = xe(Number(o(v).scale || 1) * y, 0.01, 20);
    let w = Object.assign(Object.assign({}, o(v)), { scale: M });
    const P = Ae(m.x, m.y, w);
    w = Object.assign(Object.assign({}, w), {
      center_x: Number(w.center_x || 0) + f.x - P.x,
      center_y: Number(w.center_y || 0) + f.y - P.y
    }), $e(w), qt("canvas", "滚轮缩放已同步: scale=" + M.toFixed(3));
  }
  function We() {
    ce() && ($e(Object.assign(Object.assign({}, o(v)), {
      center_x: se() / 2,
      center_y: Te() / 2,
      scale: 1,
      rotation_deg: 0
    })), Xe("reset", "居中归一 " + (V() ? fe() : "版图") + "：scale=1，rotation=0"));
  }
  function Wr() {
    ce() && ($e(Object.assign(Object.assign({}, o(v)), { center_x: se() / 2, center_y: Te() / 2 })), Xe("center", "居中 " + (V() ? fe() : "版图") + "：保留缩放和旋转"));
  }
  function ua() {
    if (!ce()) return;
    const [s, f, m, y] = ar(), M = Math.max(1, m - s + 1), w = Math.max(1, y - f + 1), P = xe(Math.min(se() / M, Te() / w) * 0.9, 0.01, 20);
    $e(Object.assign(Object.assign({}, o(v)), {
      center_x: se() / 2,
      center_y: Te() / 2,
      scale: P
    })), Xe("fit", "适配 " + (V() ? fe() : "版图") + "：scale=" + P.toFixed(3));
  }
  function fa() {
    if (!ce()) return;
    const s = Ht(), f = Math.min(...s.map((R) => R.x)), m = Math.max(...s.map((R) => R.x)), y = Math.min(...s.map((R) => R.y)), M = Math.max(...s.map((R) => R.y));
    let w = 0, P = 0;
    const C = Math.max(20, se() * 0.03);
    if (m < C ? w = C - m : f > se() - C && (w = se() - C - f), M < C ? P = C - M : y > Te() - C && (P = Te() - C - y), w === 0 && P === 0) {
      S(x, (V() ? fe() : "版图") + " 已经在视野内");
      return;
    }
    $e(Object.assign(Object.assign({}, o(v)), {
      center_x: Number(o(v).center_x || 0) + w,
      center_y: Number(o(v).center_y || 0) + P
    })), Xe("bring_into_view", "已找回 " + (V() ? fe() : "版图") + " 到视野内");
  }
  {
    let s = Re(() => o(E) || o(T) ? "focus" : "base");
    Ml(e, {
      get visible() {
        return l.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return o(s);
      },
      padding: !1,
      get elem_id() {
        return l.shared.elem_id;
      },
      get elem_classes() {
        return l.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return l.shared.container;
      },
      get scale() {
        return l.shared.scale;
      },
      get min_width() {
        return l.shared.min_width;
      },
      children: (f, m) => {
        var y = nu(), M = Oe(y);
        Wl(M, Vs(
          {
            get autoscroll() {
              return l.shared.autoscroll;
            },
            get i18n() {
              return l.i18n;
            }
          },
          () => l.shared.loading_status,
          {
            on_clear_status: () => l.dispatch("clear_status", l.shared.loading_status)
          }
        ));
        var w = j(M, 2), P = K(w);
        {
          var C = (yt) => {
            var pr = ru(), st = j(K(pr), 2);
            Ar(st, 21, $, Sr, (Zt, Pt) => {
              var It = $l(), Yr = K(It), je = {};
              ne(
                (Jr, ot) => {
                  ye(Yr, Jr), je !== (je = ot) && (It.value = (It.__value = ot) ?? "");
                },
                [
                  () => o(Pt).label || "Label " + String(o(Pt).group_id),
                  () => q(o(Pt).group_id)
                ]
              ), D(Zt, It);
            });
            var vr;
            Li(st);
            var Zr = j(st, 2);
            {
              var da = (Zt) => {
                var Pt = tu(), It = j(K(Pt), 2);
                Ar(It, 21, $, Sr, (Yr, je, Jr) => {
                  var ot = eu();
                  let Gn;
                  var gr = K(ot);
                  wt(gr, "draggable", !0);
                  var Dn = j(gr, 2);
                  Dn.textContent = Jr + 1;
                  var va = j(Dn, 2), ga = K(va);
                  ne(
                    (lt, ma) => {
                      Gn = mt(ot, 1, "group-reorder-item svelte-r41nsf", null, Gn, lt), ye(ga, ma);
                    },
                    [
                      () => ({
                        active: q(o(je).group_id) === o(b),
                        "drop-before": o(J) === q(o(je).group_id) && o(Ce) === "before",
                        "drop-after": o(J) === q(o(je).group_id) && o(Ce) === "after"
                      }),
                      () => o(je).label || "Label " + String(o(je).group_id)
                    ]
                  ), _e("dragstart", gr, (lt) => at(lt, q(o(je).group_id))), _e("dragend", gr, fr), _e("click", ot, (lt) => Vr(lt, q(o(je).group_id))), _e("dragover", ot, (lt) => bt(lt, q(o(je).group_id))), _e("drop", ot, (lt) => _t(lt, q(o(je).group_id))), D(Yr, ot);
                }), D(Zt, Pt);
              }, pa = Re(() => ve());
              re(Zr, (Zt) => {
                o(pa) && Zt(da);
              });
            }
            ne(() => {
              vr !== (vr = o(b)) && (st.value = (st.__value = o(b)) ?? "", Hr(st, o(b)));
            }), _e("change", st, hr), D(yt, pr);
          }, R = Re(() => V());
          re(P, (yt) => {
            o(R) && yt(C);
          });
        }
        var Z = j(P, 2), z = K(Z), Ee = j(z, 2), te = j(Ee, 2), Le = j(te, 2), tt = j(Z, 2), Me = K(tt);
        Ln(Me, (yt) => u = yt, () => u);
        var ca = j(tt, 2), ha = K(ca);
        ne(
          (yt, pr, st, vr, Zr) => {
            Ye(w, yt), z.disabled = pr, Ee.disabled = st, te.disabled = vr, Le.disabled = Zr, Ye(Me, "cursor:" + o(I)), ye(ha, o(x));
          },
          [
            () => "min-height:" + pe(l.props.height),
            () => !ce(),
            () => !ce(),
            () => !ce(),
            () => !ce()
          ]
        ), _e("click", z, We), _e("click", Ee, Wr), _e("click", te, ua), _e("click", Le, fa), _e("pointerdown", Me, zr), _e("pointermove", Me, Xr), _e("pointerup", Me, dr), _e("pointercancel", Me, dr), _e("pointerleave", Me, qr), _e("wheel", Me, be), D(f, y);
      },
      $$slots: { default: !0 }
    });
  }
  Br();
}
export {
  au as default
};
