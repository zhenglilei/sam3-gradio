import { i as bn, g as Pi, o as Os, n as Ct, u as he, s as Hs, r as an, q as Bi, c as Ci, a as xt, b as Ir, d as Tr, e as Is, T as Ns, f as Ps, h as wr, j as Et, p as mn, k as St, l as Li, m as Ri, t as Gi, v as Lt, w as Di, x as Rt, y as _n, E as Ui, z as a, A as ai, B as Bs, C as Fi, D as ki, F as yn, G as Cs, H as si, I as Ls, J as Rs, K as Pt, L as ze, M as ji, N as zr, O as Gs, P as Ds, Q as Us, R as Fs, S as Vi, U as ks, V as Xi, W as xn, X as oi, Y as ui, Z as js, _ as Vs, $ as Xs, a0 as Ws, a1 as zs, a2 as Nr, a3 as Ys, a4 as qs, a5 as Zs, a6 as Js, a7 as $s, a8 as En, a9 as Ks, aa as Qs, ab as eo, ac as to, ad as ro, ae as no, af as Wi, ag as io, ah as ao, ai as Ne, aj as sn, ak as so, al as oo, am as uo, an as ft, ao as _, ap as lo, aq as fo, ar as co, as as ho, at as po, au as vo, av as Z, aw as go, ax as li, ay as bo, az as Yr, aA as zi, aB as Yi, aC as z, aD as j, aE as Ke, aF as mo, aG as fi, aH as $, aI as _t, aJ as ye } from "./render-DlPxDIpa.js";
function qi(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const _o = [];
function yo(e, t = !1, r = !1) {
  return xr(e, /* @__PURE__ */ new Map(), "", _o, null, r);
}
function xr(e, t, r, n, s = null, u = !1) {
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
    if (bn(e)) {
      var c = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, c), s !== null && t.set(s, c);
      for (var d = 0; d < e.length; d += 1) {
        var h = e[d];
        d in e && (c[d] = xr(h, t, r, n, null, u));
      }
      return c;
    }
    if (Pi(e) === Os) {
      c = {}, t.set(e, c), s !== null && t.set(s, c);
      for (var p of Object.keys(e))
        c[p] = xr(
          // @ts-expect-error
          e[p],
          t,
          r,
          n,
          null,
          u
        );
      return c;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !u)
      return xr(
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
function xo(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), Ct;
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
function Eo(e, t) {
  return {
    subscribe: Pr(e, t).subscribe
  };
}
function Pr(e, t = Ct) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function s(c) {
    if (Hs(e, c) && (e = c, r)) {
      const d = !Nt.length;
      for (const h of n)
        h[1](), Nt.push(h, e);
      if (d) {
        for (let h = 0; h < Nt.length; h += 2)
          Nt[h][0](Nt[h + 1]);
        Nt.length = 0;
      }
    }
  }
  function u(c) {
    s(c(
      /** @type {T} */
      e
    ));
  }
  function l(c, d = Ct) {
    const h = [c, d];
    return n.add(h), n.size === 1 && (r = t(s, u) || Ct), c(
      /** @type {T} */
      e
    ), () => {
      n.delete(h), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: s, update: u, subscribe: l };
}
function Ft(e, t, r) {
  const n = !Array.isArray(e), s = n ? [e] : e;
  if (!s.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const u = t.length < 2;
  return Eo(r, (l, c) => {
    let d = !1;
    const h = [];
    let p = 0, x = Ct;
    const y = () => {
      if (p)
        return;
      x();
      const H = t(n ? h[0] : h, l, c);
      u ? l(H) : x = typeof H == "function" ? H : Ct;
    }, B = s.map(
      (H, P) => xo(
        H,
        (O) => {
          h[P] = O, p &= ~(1 << P), d && y();
        },
        () => {
          p |= 1 << P;
        }
      )
    );
    return d = !0, y(), function() {
      an(B), x(), d = !1;
    };
  });
}
let br = !1;
function So(e) {
  var t = br;
  try {
    return br = !1, [e(), br];
  } finally {
    br = t;
  }
}
function To(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Bi(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const wo = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Ao(e) {
  return (
    /** @type {string} */
    wo?.createHTML(e) ?? e
  );
}
function Zi(e) {
  var t = Ci("template");
  return t.innerHTML = Ao(e.replaceAll("<!>", "<!---->")), t.content;
}
function nr(e, t) {
  var r = (
    /** @type {Effect} */
    Ir
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function xe(e, t) {
  var r = (t & Ns) !== 0, n = (t & Ps) !== 0, s, u = !e.startsWith("<!>");
  return () => {
    s === void 0 && (s = Zi(u ? e : "<!>" + e), r || (s = /** @type {TemplateNode} */
    Tr(s)));
    var l = (
      /** @type {TemplateNode} */
      n || Is ? document.importNode(s, !0) : s.cloneNode(!0)
    );
    if (r) {
      var c = (
        /** @type {TemplateNode} */
        Tr(l)
      ), d = (
        /** @type {TemplateNode} */
        l.lastChild
      );
      nr(c, d);
    } else
      nr(l, l);
    return l;
  };
}
// @__NO_SIDE_EFFECTS__
function Mo(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), s = `<${r}>${n ? e : "<!>" + e}</${r}>`, u;
  return () => {
    if (!u) {
      var l = (
        /** @type {DocumentFragment} */
        Zi(s)
      ), c = (
        /** @type {Element} */
        Tr(l)
      );
      u = /** @type {Element} */
      Tr(c);
    }
    var d = (
      /** @type {TemplateNode} */
      u.cloneNode(!0)
    );
    return nr(d, d), d;
  };
}
// @__NO_SIDE_EFFECTS__
function Oo(e, t) {
  return /* @__PURE__ */ Mo(e, t, "svg");
}
function Ho() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = xt();
  return e.append(t, r), nr(t, r), e;
}
function fe(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Ji {
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
        wr(n), this.#n.delete(r);
      else {
        var s = this.#e.get(r);
        s && (wr(s.effect), this.#r.set(r, s.effect), this.#e.delete(r), s.fragment.lastChild.remove(), this.anchor.before(s.fragment), n = s.effect);
      }
      for (const [u, l] of this.#t) {
        if (this.#t.delete(u), u === t)
          break;
        const c = this.#e.get(l);
        c && (Et(c.effect), this.#e.delete(l));
      }
      for (const [u, l] of this.#r) {
        if (u === r || this.#n.has(u)) continue;
        const c = () => {
          if (Array.from(this.#t.values()).includes(u)) {
            var h = document.createDocumentFragment();
            Ri(l, h), h.append(xt()), this.#e.set(u, { effect: l, fragment: h });
          } else
            Et(l);
          this.#n.delete(u), this.#r.delete(u);
        };
        this.#i || !n ? (this.#n.add(u), mn(l, c, !1)) : c();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #s = (t) => {
    this.#t.delete(t);
    const r = Array.from(this.#t.values());
    for (const [n, s] of this.#e)
      r.includes(n) || (Et(s.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Li
    ), s = Gi();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (s) {
        var u = document.createDocumentFragment(), l = xt();
        u.append(l), this.#e.set(t, {
          effect: St(() => r(l)),
          fragment: u
        });
      } else
        this.#r.set(
          t,
          St(() => r(this.anchor))
        );
    if (this.#t.set(n, t), s) {
      for (const [c, d] of this.#r)
        c === t ? n.unskip_effect(d) : n.skip_effect(d);
      for (const [c, d] of this.#e)
        c === t ? n.unskip_effect(d.effect) : n.skip_effect(d.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Io(e) {
  Lt === null && qi(), Di && Lt.l !== null ? Po(Lt).m.push(e) : Rt(() => {
    const t = he(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function No(e) {
  Lt === null && qi(), Io(() => () => he(e));
}
function Po(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function We(e, t, r = !1) {
  var n = new Ji(e), s = r ? Ui : 0;
  function u(l, c) {
    n.ensure(l, c);
  }
  _n(() => {
    var l = !1;
    t((c, d = 0) => {
      l = !0, u(d, c);
    }), l || u(-1, null);
  }, s);
}
function mr(e, t) {
  return t;
}
function Bo(e, t, r) {
  for (var n = [], s = t.length, u, l = t.length, c = 0; c < s; c++) {
    let x = t[c];
    mn(
      x,
      () => {
        if (u) {
          if (u.pending.delete(x), u.done.add(x), u.pending.size === 0) {
            var y = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            on(e, yn(u.done)), y.delete(u), y.size === 0 && (e.outrogroups = null);
          }
        } else
          l -= 1;
      },
      !1
    );
  }
  if (l === 0) {
    var d = n.length === 0 && r !== null;
    if (d) {
      var h = (
        /** @type {Element} */
        r
      ), p = (
        /** @type {Element} */
        h.parentNode
      );
      Us(p), p.append(h), e.items.clear();
    }
    on(e, t, !d);
  } else
    u = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(u);
}
function on(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const l of e.pending.values())
      for (const c of l)
        n.add(
          /** @type {EachItem} */
          e.items.get(c).e
        );
  }
  for (var s = 0; s < t.length; s++) {
    var u = t[s];
    if (n?.has(u)) {
      u.f |= ze;
      const l = document.createDocumentFragment();
      Ri(u, l);
    } else
      Et(t[s], r);
  }
}
var ci;
function _r(e, t, r, n, s, u = null) {
  var l = e, c = /* @__PURE__ */ new Map(), d = (t & Fi) !== 0;
  if (d) {
    var h = (
      /** @type {Element} */
      e
    );
    l = h.appendChild(xt());
  }
  var p = null, x = ki(() => {
    var T = r();
    return (
      /** @type {V[]} */
      bn(T) ? T : T == null ? [] : yn(T)
    );
  }), y, B = /* @__PURE__ */ new Map(), H = !0;
  function P(T) {
    (A.effect.f & ji) === 0 && (A.pending.delete(T), A.fallback = p, Co(A, y, l, t, n), p !== null && (y.length === 0 ? (p.f & ze) === 0 ? wr(p) : (p.f ^= ze, er(p, null, l)) : mn(p, () => {
      p = null;
    })));
  }
  function O(T) {
    A.pending.delete(T);
  }
  var E = _n(() => {
    y = /** @type {V[]} */
    a(x);
    for (var T = y.length, g = /* @__PURE__ */ new Set(), b = (
      /** @type {Batch} */
      Li
    ), M = Gi(), C = 0; C < T; C += 1) {
      var D = y[C], re = n(D, C), k = H ? null : c.get(re);
      k ? (k.v && ai(k.v, D), k.i && ai(k.i, C), M && b.unskip_effect(k.e)) : (k = Lo(
        c,
        H ? l : ci ??= xt(),
        D,
        re,
        C,
        s,
        t,
        r
      ), H || (k.e.f |= ze), c.set(re, k)), g.add(re);
    }
    if (T === 0 && u && !p && (H ? p = St(() => u(l)) : (p = St(() => u(ci ??= xt())), p.f |= ze)), T > g.size && Bs(), !H)
      if (B.set(b, g), M) {
        for (const [Le, Re] of c)
          g.has(Le) || b.skip_effect(Re.e);
        b.oncommit(P), b.ondiscard(O);
      } else
        P(b);
    a(x);
  }), A = { effect: E, items: c, pending: B, outrogroups: null, fallback: p };
  H = !1;
}
function Kt(e) {
  for (; e !== null && (e.f & Gs) === 0; )
    e = e.next;
  return e;
}
function Co(e, t, r, n, s) {
  var u = (n & Ds) !== 0, l = t.length, c = e.items, d = Kt(e.effect.first), h, p = null, x, y = [], B = [], H, P, O, E;
  if (u)
    for (E = 0; E < l; E += 1)
      H = t[E], P = s(H, E), O = /** @type {EachItem} */
      c.get(P).e, (O.f & ze) === 0 && (O.nodes?.a?.measure(), (x ??= /* @__PURE__ */ new Set()).add(O));
  for (E = 0; E < l; E += 1) {
    if (H = t[E], P = s(H, E), O = /** @type {EachItem} */
    c.get(P).e, e.outrogroups !== null)
      for (const k of e.outrogroups)
        k.pending.delete(O), k.done.delete(O);
    if ((O.f & zr) !== 0 && (wr(O), u && (O.nodes?.a?.unfix(), (x ??= /* @__PURE__ */ new Set()).delete(O))), (O.f & ze) !== 0)
      if (O.f ^= ze, O === d)
        er(O, null, r);
      else {
        var A = p ? p.next : d;
        O === e.effect.last && (e.effect.last = O.prev), O.prev && (O.prev.next = O.next), O.next && (O.next.prev = O.prev), lt(e, p, O), lt(e, O, A), er(O, A, r), p = O, y = [], B = [], d = Kt(p.next);
        continue;
      }
    if (O !== d) {
      if (h !== void 0 && h.has(O)) {
        if (y.length < B.length) {
          var T = B[0], g;
          p = T.prev;
          var b = y[0], M = y[y.length - 1];
          for (g = 0; g < y.length; g += 1)
            er(y[g], T, r);
          for (g = 0; g < B.length; g += 1)
            h.delete(B[g]);
          lt(e, b.prev, M.next), lt(e, p, b), lt(e, M, T), d = T, p = M, E -= 1, y = [], B = [];
        } else
          h.delete(O), er(O, d, r), lt(e, O.prev, O.next), lt(e, O, p === null ? e.effect.first : p.next), lt(e, p, O), p = O;
        continue;
      }
      for (y = [], B = []; d !== null && d !== O; )
        (h ??= /* @__PURE__ */ new Set()).add(d), B.push(d), d = Kt(d.next);
      if (d === null)
        continue;
    }
    (O.f & ze) === 0 && y.push(O), p = O, d = Kt(O.next);
  }
  if (e.outrogroups !== null) {
    for (const k of e.outrogroups)
      k.pending.size === 0 && (on(e, yn(k.done)), e.outrogroups?.delete(k));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (d !== null || h !== void 0) {
    var C = [];
    if (h !== void 0)
      for (O of h)
        (O.f & zr) === 0 && C.push(O);
    for (; d !== null; )
      (d.f & zr) === 0 && d !== e.fallback && C.push(d), d = Kt(d.next);
    var D = C.length;
    if (D > 0) {
      var re = (n & Fi) !== 0 && l === 0 ? r : null;
      if (u) {
        for (E = 0; E < D; E += 1)
          C[E].nodes?.a?.measure();
        for (E = 0; E < D; E += 1)
          C[E].nodes?.a?.fix();
      }
      Bo(e, C, re);
    }
  }
  u && Bi(() => {
    if (x !== void 0)
      for (O of x)
        O.nodes?.a?.apply();
  });
}
function Lo(e, t, r, n, s, u, l, c) {
  var d = (l & Ls) !== 0 ? (l & Rs) === 0 ? Pt(r, !1, !1) : si(r) : null, h = (l & Cs) !== 0 ? si(s) : null;
  return {
    v: d,
    i: h,
    e: St(() => (u(t, d ?? r, h ?? s, c), () => {
      e.delete(n);
    }))
  };
}
function er(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, s = e.nodes.end, u = t && (t.f & ze) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var l = (
        /** @type {TemplateNode} */
        Fs(n)
      );
      if (u.before(n), n === s)
        return;
      n = l;
    }
}
function lt(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Ro(e, t, r, n, s) {
  var u = t.$$slots?.[r], l = !1;
  u === !0 && (u = t.children, l = !0), u === void 0 || u(e, l ? () => n : n);
}
function Go(e, t, r, n, s, u) {
  var l = null, c = (
    /** @type {TemplateNode} */
    e
  ), d = new Ji(c, !1);
  _n(() => {
    const h = t() || null;
    var p = h === "svg" ? ks : void 0;
    if (h === null) {
      d.ensure(null, null);
      return;
    }
    return d.ensure(h, (x) => {
      if (h) {
        if (l = Ci(h, p), nr(l, l), n) {
          var y = null, B = l.appendChild(xt());
          n(l, B), y?.remove();
        }
        Ir.nodes.end = l, x.before(l);
      }
    }), () => {
    };
  }, Ui), Vi(() => {
  });
}
function Do(e, t) {
  var r = void 0, n;
  Xi(() => {
    r !== (r = t()) && (n && (Et(n), n = null), r && (n = St(() => {
      xn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function $i(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var s = e.length;
    for (t = 0; t < s; t++) e[t] && (r = $i(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Uo() {
  for (var e, t, r = 0, n = "", s = arguments.length; r < s; r++) (e = arguments[r]) && (t = $i(e)) && (n && (n += " "), n += t);
  return n;
}
function Fo(e) {
  return typeof e == "object" ? Uo(e) : e ?? "";
}
const hi = [...` 	
\r\f \v\uFEFF`];
function ko(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var s of Object.keys(r))
      if (r[s])
        n = n ? n + " " + s : s;
      else if (n.length)
        for (var u = s.length, l = 0; (l = n.indexOf(s, l)) >= 0; ) {
          var c = l + u;
          (l === 0 || hi.includes(n[l - 1])) && (c === n.length || hi.includes(n[c])) ? n = (l === 0 ? "" : n.substring(0, l)) + n.substring(c + 1) : l = c;
        }
  }
  return n === "" ? null : n;
}
function di(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var s of Object.keys(e)) {
    var u = e[s];
    u != null && u !== "" && (n += " " + s + ": " + u + r);
  }
  return n;
}
function qr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function jo(e, t) {
  if (t) {
    var r = "", n, s;
    if (Array.isArray(t) ? (n = t[0], s = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var u = !1, l = 0, c = !1, d = [];
      n && d.push(...Object.keys(n).map(qr)), s && d.push(...Object.keys(s).map(qr));
      var h = 0, p = -1;
      const P = e.length;
      for (var x = 0; x < P; x++) {
        var y = e[x];
        if (c ? y === "/" && e[x - 1] === "*" && (c = !1) : u ? u === y && (u = !1) : y === "/" && e[x + 1] === "*" ? c = !0 : y === '"' || y === "'" ? u = y : y === "(" ? l++ : y === ")" && l--, !c && u === !1 && l === 0) {
          if (y === ":" && p === -1)
            p = x;
          else if (y === ";" || x === P - 1) {
            if (p !== -1) {
              var B = qr(e.substring(h, p).trim());
              if (!d.includes(B)) {
                y !== ";" && x++;
                var H = e.substring(h, x).trim();
                r += " " + H + ";";
              }
            }
            h = x + 1, p = -1;
          }
        }
      }
    }
    return n && (r += di(n)), s && (r += di(s, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function yt(e, t, r, n, s, u) {
  var l = (
    /** @type {any} */
    e[oi]
  );
  if (l !== r || l === void 0) {
    var c = ko(r, n, u);
    c == null ? e.removeAttribute("class") : t ? e.className = c : e.setAttribute("class", c), e[oi] = r;
  } else if (u && s !== u)
    for (var d in u) {
      var h = !!u[d];
      (s == null || h !== !!s[d]) && e.classList.toggle(d, h);
    }
  return u;
}
function Zr(e, t = {}, r, n) {
  for (var s in r) {
    var u = r[s];
    t[s] !== u && (r[s] == null ? e.style.removeProperty(s) : e.style.setProperty(s, u, n));
  }
}
function Ar(e, t, r, n) {
  var s = (
    /** @type {any} */
    e[ui]
  );
  if (s !== t) {
    var u = jo(t, n);
    u == null ? e.removeAttribute("style") : e.style.cssText = u, e[ui] = t;
  } else n && (Array.isArray(n) ? (Zr(e, r?.[0], n[0]), Zr(e, r?.[1], n[1], "important")) : Zr(e, r, n));
  return n;
}
function Mr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!bn(t))
      return js();
    for (var n of e.options)
      n.selected = t.includes(pi(n));
    return;
  }
  for (n of e.options) {
    var s = pi(n);
    if (Vs(s, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Ki(e) {
  var t = new MutationObserver(() => {
    Mr(e, e.__value);
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
  }), Vi(() => {
    t.disconnect();
  });
}
function pi(e) {
  return "__value" in e ? e.__value : e.value;
}
const tr = /* @__PURE__ */ Symbol("class"), Bt = /* @__PURE__ */ Symbol("style"), Qi = /* @__PURE__ */ Symbol("is custom element"), ea = /* @__PURE__ */ Symbol("is html"), Vo = Nr ? "input" : "INPUT", Xo = Nr ? "option" : "OPTION", Wo = Nr ? "select" : "SELECT", zo = Nr ? "progress" : "PROGRESS";
function Yo(e, t) {
  var r = Sn(e);
  r.value === (r.value = // treat null and undefined the same for the initial value
  t ?? void 0) || // @ts-expect-error
  // `progress` elements always need their value set when it's `0`
  e.value === t && (t !== 0 || e.nodeName !== zo) || (e.value = t ?? "");
}
function qo(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function rr(e, t, r, n) {
  var s = Sn(e);
  s[t] !== (s[t] = r) && (t === "loading" && (e[Xs] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ta(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Zo(e, t, r, n, s = !1, u = !1) {
  var l = Sn(e), c = l[Qi], d = !l[ea], h = t || {}, p = e.nodeName === Xo;
  for (var x in t)
    x in r || (r[x] = null);
  r.class ? r.class = Fo(r.class) : r.class = null, r[Bt] && (r.style ??= null);
  var y = ta(e);
  if (e.nodeName === Vo && "type" in r && ("value" in r || "__value" in r)) {
    var B = r.type;
    (B !== h.type || B === void 0 && e.hasAttribute("type")) && (h.type = B, rr(e, "type", B));
  }
  for (const g in r) {
    let b = r[g];
    if (p && g === "value" && b == null) {
      e.value = e.__value = "", h[g] = b;
      continue;
    }
    if (g === "class") {
      var H = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      yt(e, H, b, n, t?.[tr], r[tr]), h[g] = b, h[tr] = r[tr];
      continue;
    }
    if (g === "style") {
      Ar(e, b, t?.[Bt], r[Bt]), h[g] = b, h[Bt] = r[Bt];
      continue;
    }
    var P = h[g];
    if (!(b === P && !(b === void 0 && e.hasAttribute(g)))) {
      h[g] = b;
      var O = g[0] + g[1];
      if (O !== "$$")
        if (O === "on") {
          const M = {}, C = "$$" + g;
          let D = g.slice(2);
          var E = to(D);
          if (Js(D) && (D = D.slice(0, -7), M.capture = !0), !E && P) {
            if (b != null) continue;
            e.removeEventListener(D, h[C], M), h[C] = null;
          }
          if (E)
            $s(D, e, b), En([D]);
          else if (b != null) {
            let re = function(k) {
              h[g].call(this, k);
            };
            h[C] = Ks(D, e, re, M);
          }
        } else if (g === "style")
          rr(e, g, b);
        else if (g === "autofocus")
          To(
            /** @type {HTMLElement} */
            e,
            !!b
          );
        else if (!c && (g === "__value" || g === "value" && b != null))
          e.value = e.__value = b;
        else if (g === "selected" && p)
          qo(
            /** @type {HTMLOptionElement} */
            e,
            b
          );
        else {
          var A = g;
          d || (A = Qs(A));
          var T = A === "defaultValue" || A === "defaultChecked";
          if (b == null && !c && !T)
            if (l[g] = null, A === "value" || A === "checked") {
              let M = (
                /** @type {HTMLInputElement} */
                e
              );
              const C = t === void 0;
              if (A === "value") {
                let D = M.defaultValue;
                M.removeAttribute(A), M.defaultValue = D, M.value = M.__value = C ? D : null;
              } else {
                let D = M.defaultChecked;
                M.removeAttribute(A), M.defaultChecked = D, M.checked = C ? D : !1;
              }
            } else
              e.removeAttribute(g);
          else T || y.includes(A) && (c || typeof b != "string") ? (e[A] = b, A in l && (l[A] = eo)) : typeof b != "function" && rr(e, A, b);
        }
    }
  }
  return h;
}
function Jo(e, t, r = [], n = [], s = [], u, l = !1, c = !1) {
  qs(s, r, n, (d) => {
    var h = void 0, p = {}, x = e.nodeName === Wo, y = !1;
    if (Xi(() => {
      var H = t(...d.map(a)), P = Zo(
        e,
        h,
        H,
        u,
        l,
        c
      );
      y && x && "value" in H && Mr(
        /** @type {HTMLSelectElement} */
        e,
        H.value
      );
      for (let E of Object.getOwnPropertySymbols(p))
        H[E] || Et(p[E]);
      for (let E of Object.getOwnPropertySymbols(H)) {
        var O = H[E];
        E.description === Zs && (!h || O !== h[E]) && (p[E] && Et(p[E]), p[E] = St(() => Do(e, () => O))), P[E] = O;
      }
      h = P;
    }), x) {
      var B = (
        /** @type {HTMLSelectElement} */
        e
      );
      xn(() => {
        Mr(
          B,
          /** @type {Record<string | symbol, any>} */
          h.value,
          !0
        ), Ki(B);
      });
    }
    y = !0;
  });
}
function Sn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Ws] ??= {
      [Qi]: e.nodeName.includes("-"),
      [ea]: e.namespaceURI === zs
    }
  );
}
var vi = /* @__PURE__ */ new Map();
function ta(e) {
  var t = e.getAttribute("is") || e.nodeName, r = vi.get(t);
  if (r) return r;
  vi.set(t, r = []);
  for (var n, s = e, u = Element.prototype; u !== s; ) {
    n = Ys(s);
    for (var l in n)
      n[l].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      l !== "innerHTML" && l !== "textContent" && l !== "innerText" && r.push(l);
    s = Pi(s);
  }
  return r;
}
function Jr(e, t) {
  return e === t || e?.[Wi] === t;
}
function ra(e = {}, t, r, n) {
  var s = (
    /** @type {ComponentContext} */
    Lt.r
  ), u = (
    /** @type {Effect} */
    Ir
  );
  return xn(() => {
    var l, c;
    return ro(() => {
      l = c, c = [], he(() => {
        Jr(r(...c), e) || (t(e, ...c), l && Jr(r(...l), e) && t(null, ...l));
      });
    }), () => {
      let d = u;
      for (; d !== s && d.parent !== null && d.parent.f & no; )
        d = d.parent;
      const h = () => {
        c && Jr(r(...c), e) && t(null, ...c);
      }, p = d.teardown;
      d.teardown = () => {
        h(), p?.();
      };
    };
  }), e;
}
function $o(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Lt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Ne(t.s);
  if (e) {
    let s = 0, u = (
      /** @type {Record<string, any>} */
      {}
    );
    const l = sn(() => {
      let c = !1;
      const d = t.s;
      for (const h in d)
        d[h] !== u[h] && (u[h] = d[h], c = !0);
      return c && s++, s;
    });
    n = () => a(l);
  }
  r.b.length && io(() => {
    gi(t, n), an(r.b);
  }), Rt(() => {
    const s = he(() => r.m.map(ao));
    return () => {
      for (const u of s)
        typeof u == "function" && u();
    };
  }), r.a.length && Rt(() => {
    gi(t, n), an(r.a);
  });
}
function gi(e, t) {
  if (e.l.s)
    for (const r of e.l.s) a(r);
  t();
}
const Ko = {
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
function Qo(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Ko
  );
}
function te(e, t, r, n) {
  var s = !Di || (r & fo) !== 0, u = (r & lo) !== 0, l = (r & ho) !== 0, c = (
    /** @type {V} */
    n
  ), d = !0, h = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), p = () => l && s ? (h ??= sn(
    /** @type {() => V} */
    n
  ), a(h)) : (d && (d = !1, c = l ? he(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), c);
  let x;
  if (u) {
    var y = Wi in e || vo in e;
    x = so(e, t)?.set ?? (y && t in e ? (g) => e[t] = g : void 0);
  }
  var B, H = !1;
  u ? [B, H] = So(() => (
    /** @type {V} */
    e[t]
  )) : B = /** @type {V} */
  e[t], B === void 0 && n !== void 0 && (B = p(), x && (s && oo(), x(B)));
  var P;
  if (s ? P = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g === void 0 ? p() : (d = !0, g);
  } : P = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g !== void 0 && (c = /** @type {V} */
    void 0), g === void 0 ? c : g;
  }, s && (r & uo) === 0)
    return P;
  if (x) {
    var O = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(g, b) {
        return arguments.length > 0 ? ((!s || !b || O || H) && x(b ? P() : g), g) : P();
      })
    );
  }
  var E = !1, A = ((r & co) !== 0 ? sn : ki)(() => (E = !1, P()));
  u && a(A);
  var T = (
    /** @type {Effect} */
    Ir
  );
  return (
    /** @type {() => V} */
    (function(g, b) {
      if (arguments.length > 0) {
        const M = b ? a(A) : s && u ? ft(g) : g;
        return _(A, M), E = !0, c !== void 0 && (c = M), g;
      }
      return po && E || (T.f & ji) !== 0 ? A.v : a(A);
    })
  );
}
const eu = [
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
], bi = {
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
eu.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: bi[t][r],
    secondary: bi[t][n]
  }
}), {});
function tu(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var $r, mi;
function ru() {
  if (mi) return $r;
  mi = 1;
  var e = function(A) {
    return t(A) && !r(A);
  };
  function t(E) {
    return !!E && typeof E == "object";
  }
  function r(E) {
    var A = Object.prototype.toString.call(E);
    return A === "[object RegExp]" || A === "[object Date]" || u(E);
  }
  var n = typeof Symbol == "function" && Symbol.for, s = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function u(E) {
    return E.$$typeof === s;
  }
  function l(E) {
    return Array.isArray(E) ? [] : {};
  }
  function c(E, A) {
    return A.clone !== !1 && A.isMergeableObject(E) ? P(l(E), E, A) : E;
  }
  function d(E, A, T) {
    return E.concat(A).map(function(g) {
      return c(g, T);
    });
  }
  function h(E, A) {
    if (!A.customMerge)
      return P;
    var T = A.customMerge(E);
    return typeof T == "function" ? T : P;
  }
  function p(E) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(E).filter(function(A) {
      return Object.propertyIsEnumerable.call(E, A);
    }) : [];
  }
  function x(E) {
    return Object.keys(E).concat(p(E));
  }
  function y(E, A) {
    try {
      return A in E;
    } catch {
      return !1;
    }
  }
  function B(E, A) {
    return y(E, A) && !(Object.hasOwnProperty.call(E, A) && Object.propertyIsEnumerable.call(E, A));
  }
  function H(E, A, T) {
    var g = {};
    return T.isMergeableObject(E) && x(E).forEach(function(b) {
      g[b] = c(E[b], T);
    }), x(A).forEach(function(b) {
      B(E, b) || (y(E, b) && T.isMergeableObject(A[b]) ? g[b] = h(b, T)(E[b], A[b], T) : g[b] = c(A[b], T));
    }), g;
  }
  function P(E, A, T) {
    T = T || {}, T.arrayMerge = T.arrayMerge || d, T.isMergeableObject = T.isMergeableObject || e, T.cloneUnlessOtherwiseSpecified = c;
    var g = Array.isArray(A), b = Array.isArray(E), M = g === b;
    return M ? g ? T.arrayMerge(E, A, T) : H(E, A, T) : c(A, T);
  }
  P.all = function(A, T) {
    if (!Array.isArray(A))
      throw new Error("first argument should be an array");
    return A.reduce(function(g, b) {
      return P(g, b, T);
    }, {});
  };
  var O = P;
  return $r = O, $r;
}
var nu = ru();
const iu = /* @__PURE__ */ tu(nu);
var un = function(e, t) {
  return un = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var s in n) Object.prototype.hasOwnProperty.call(n, s) && (r[s] = n[s]);
  }, un(e, t);
};
function Br(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  un(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var V = function() {
  return V = Object.assign || function(t) {
    for (var r, n = 1, s = arguments.length; n < s; n++) {
      r = arguments[n];
      for (var u in r) Object.prototype.hasOwnProperty.call(r, u) && (t[u] = r[u]);
    }
    return t;
  }, V.apply(this, arguments);
};
function au(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var s = 0, n = Object.getOwnPropertySymbols(e); s < n.length; s++)
      t.indexOf(n[s]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[s]) && (r[n[s]] = e[n[s]]);
  return r;
}
function Kr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, s = t.length, u; n < s; n++)
    (u || !(n in t)) && (u || (u = Array.prototype.slice.call(t, 0, n)), u[n] = t[n]);
  return e.concat(u || Array.prototype.slice.call(t));
}
function Qr(e, t) {
  var r = t && t.cache ? t.cache : hu, n = t && t.serializer ? t.serializer : fu, s = t && t.strategy ? t.strategy : uu;
  return s(e, {
    cache: r,
    serializer: n
  });
}
function su(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function ou(e, t, r, n) {
  var s = su(n) ? n : r(n), u = t.get(s);
  return typeof u > "u" && (u = e.call(this, n), t.set(s, u)), u;
}
function na(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), s = r(n), u = t.get(s);
  return typeof u > "u" && (u = e.apply(this, n), t.set(s, u)), u;
}
function ia(e, t, r, n, s) {
  return r.bind(t, e, n, s);
}
function uu(e, t) {
  var r = e.length === 1 ? ou : na;
  return ia(e, this, r, t.cache.create(), t.serializer);
}
function lu(e, t) {
  return ia(e, this, na, t.cache.create(), t.serializer);
}
var fu = function() {
  return JSON.stringify(arguments);
}, cu = (
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
), hu = {
  create: function() {
    return new cu();
  }
}, en = {
  variadic: lu
}, G;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(G || (G = {}));
var K;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(K || (K = {}));
var Gt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(Gt || (Gt = {}));
function _i(e) {
  return e.type === K.literal;
}
function du(e) {
  return e.type === K.argument;
}
function aa(e) {
  return e.type === K.number;
}
function sa(e) {
  return e.type === K.date;
}
function oa(e) {
  return e.type === K.time;
}
function ua(e) {
  return e.type === K.select;
}
function la(e) {
  return e.type === K.plural;
}
function pu(e) {
  return e.type === K.pound;
}
function fa(e) {
  return e.type === K.tag;
}
function ca(e) {
  return !!(e && typeof e == "object" && e.type === Gt.number);
}
function ln(e) {
  return !!(e && typeof e == "object" && e.type === Gt.dateTime);
}
var ha = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, vu = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function gu(e) {
  var t = {};
  return e.replace(vu, function(r) {
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
var bu = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function mu(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(bu).filter(function(y) {
    return y.length > 0;
  }), r = [], n = 0, s = t; n < s.length; n++) {
    var u = s[n], l = u.split("/");
    if (l.length === 0)
      throw new Error("Invalid number skeleton");
    for (var c = l[0], d = l.slice(1), h = 0, p = d; h < p.length; h++) {
      var x = p[h];
      if (x.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: c, options: d });
  }
  return r;
}
function _u(e) {
  return e.replace(/^(.*?)-/, "");
}
var yi = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, da = /^(@+)?(\+|#+)?[rs]?$/g, yu = /(\*)(0+)|(#+)(0+)|(0+)/g, pa = /^(0+)$/;
function xi(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(da, function(r, n, s) {
    return typeof s != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : s === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof s == "string" ? s.length : 0)), "";
  }), t;
}
function va(e) {
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
function xu(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !pa.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Ei(e) {
  var t = {}, r = va(e);
  return r || t;
}
function Eu(e) {
  for (var t = {}, r = 0, n = e; r < n.length; r++) {
    var s = n[r];
    switch (s.stem) {
      case "percent":
      case "%":
        t.style = "percent";
        continue;
      case "%x100":
        t.style = "percent", t.scale = 100;
        continue;
      case "currency":
        t.style = "currency", t.currency = s.options[0];
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
        t.style = "unit", t.unit = _u(s.options[0]);
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
        t = V(V(V({}, t), { notation: "scientific" }), s.options.reduce(function(d, h) {
          return V(V({}, d), Ei(h));
        }, {}));
        continue;
      case "engineering":
        t = V(V(V({}, t), { notation: "engineering" }), s.options.reduce(function(d, h) {
          return V(V({}, d), Ei(h));
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
        t.scale = parseFloat(s.options[0]);
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
        if (s.options.length > 1)
          throw new RangeError("integer-width stems only accept a single optional option");
        s.options[0].replace(yu, function(d, h, p, x, y, B) {
          if (h)
            t.minimumIntegerDigits = p.length;
          else {
            if (x && y)
              throw new Error("We currently do not support maximum integer digits");
            if (B)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (pa.test(s.stem)) {
      t.minimumIntegerDigits = s.stem.length;
      continue;
    }
    if (yi.test(s.stem)) {
      if (s.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      s.stem.replace(yi, function(d, h, p, x, y, B) {
        return p === "*" ? t.minimumFractionDigits = h.length : x && x[0] === "#" ? t.maximumFractionDigits = x.length : y && B ? (t.minimumFractionDigits = y.length, t.maximumFractionDigits = y.length + B.length) : (t.minimumFractionDigits = h.length, t.maximumFractionDigits = h.length), "";
      });
      var u = s.options[0];
      u === "w" ? t = V(V({}, t), { trailingZeroDisplay: "stripIfInteger" }) : u && (t = V(V({}, t), xi(u)));
      continue;
    }
    if (da.test(s.stem)) {
      t = V(V({}, t), xi(s.stem));
      continue;
    }
    var l = va(s.stem);
    l && (t = V(V({}, t), l));
    var c = xu(s.stem);
    c && (t = V(V({}, t), c));
  }
  return t;
}
var yr = {
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
function Su(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var s = e.charAt(n);
    if (s === "j") {
      for (var u = 0; n + 1 < e.length && e.charAt(n + 1) === s; )
        u++, n++;
      var l = 1 + (u & 1), c = u < 2 ? 1 : 3 + (u >> 1), d = "a", h = Tu(t);
      for ((h == "H" || h == "k") && (c = 0); c-- > 0; )
        r += d;
      for (; l-- > 0; )
        r = h + r;
    } else s === "J" ? r += "H" : r += s;
  }
  return r;
}
function Tu(e) {
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
  var s = yr[n || ""] || yr[r || ""] || yr["".concat(r, "-001")] || yr["001"];
  return s[0];
}
var tn, wu = new RegExp("^".concat(ha.source, "*")), Au = new RegExp("".concat(ha.source, "*$"));
function U(e, t) {
  return { start: e, end: t };
}
var Mu = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ou = !!String.fromCodePoint, Hu = !!Object.fromEntries, Iu = !!String.prototype.codePointAt, Nu = !!String.prototype.trimStart, Pu = !!String.prototype.trimEnd, Bu = !!Number.isSafeInteger, Cu = Bu ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, fn = !0;
try {
  var Lu = ba("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  fn = ((tn = Lu.exec("a")) === null || tn === void 0 ? void 0 : tn[0]) === "a";
} catch {
  fn = !1;
}
var Si = Mu ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), cn = Ou ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", s = t.length, u = 0, l; s > u; ) {
      if (l = t[u++], l > 1114111)
        throw RangeError(l + " is not a valid code point");
      n += l < 65536 ? String.fromCharCode(l) : String.fromCharCode(((l -= 65536) >> 10) + 55296, l % 1024 + 56320);
    }
    return n;
  }
), Ti = (
  // native
  Hu ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, s = t; n < s.length; n++) {
        var u = s[n], l = u[0], c = u[1];
        r[l] = c;
      }
      return r;
    }
  )
), ga = Iu ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var s = t.charCodeAt(r), u;
      return s < 55296 || s > 56319 || r + 1 === n || (u = t.charCodeAt(r + 1)) < 56320 || u > 57343 ? s : (s - 55296 << 10) + (u - 56320) + 65536;
    }
  }
), Ru = Nu ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(wu, "");
  }
), Gu = Pu ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Au, "");
  }
);
function ba(e, t) {
  return new RegExp(e, t);
}
var hn;
if (fn) {
  var wi = ba("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  hn = function(t, r) {
    var n;
    wi.lastIndex = r;
    var s = wi.exec(t);
    return (n = s[1]) !== null && n !== void 0 ? n : "";
  };
} else
  hn = function(t, r) {
    for (var n = []; ; ) {
      var s = ga(t, r);
      if (s === void 0 || ma(s) || ku(s))
        break;
      n.push(s), r += s >= 65536 ? 2 : 1;
    }
    return cn.apply(void 0, n);
  };
var Du = (
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
      for (var s = []; !this.isEOF(); ) {
        var u = this.char();
        if (u === 123) {
          var l = this.parseArgument(t, n);
          if (l.err)
            return l;
          s.push(l.val);
        } else {
          if (u === 125 && t > 0)
            break;
          if (u === 35 && (r === "plural" || r === "selectordinal")) {
            var c = this.clonePosition();
            this.bump(), s.push({
              type: K.pound,
              location: U(c, this.clonePosition())
            });
          } else if (u === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(G.UNMATCHED_CLOSING_TAG, U(this.clonePosition(), this.clonePosition()));
          } else if (u === 60 && !this.ignoreTag && dn(this.peek() || 0)) {
            var l = this.parseTag(t, r);
            if (l.err)
              return l;
            s.push(l.val);
          } else {
            var l = this.parseLiteral(t, r);
            if (l.err)
              return l;
            s.push(l.val);
          }
        }
      }
      return { val: s, err: null };
    }, e.prototype.parseTag = function(t, r) {
      var n = this.clonePosition();
      this.bump();
      var s = this.parseTagName();
      if (this.bumpSpace(), this.bumpIf("/>"))
        return {
          val: {
            type: K.literal,
            value: "<".concat(s, "/>"),
            location: U(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var u = this.parseMessage(t + 1, r, !0);
        if (u.err)
          return u;
        var l = u.val, c = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !dn(this.char()))
            return this.error(G.INVALID_TAG, U(c, this.clonePosition()));
          var d = this.clonePosition(), h = this.parseTagName();
          return s !== h ? this.error(G.UNMATCHED_CLOSING_TAG, U(d, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: K.tag,
              value: s,
              children: l,
              location: U(n, this.clonePosition())
            },
            err: null
          } : this.error(G.INVALID_TAG, U(c, this.clonePosition())));
        } else
          return this.error(G.UNCLOSED_TAG, U(n, this.clonePosition()));
      } else
        return this.error(G.INVALID_TAG, U(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Fu(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), s = ""; ; ) {
        var u = this.tryParseQuote(r);
        if (u) {
          s += u;
          continue;
        }
        var l = this.tryParseUnquoted(t, r);
        if (l) {
          s += l;
          continue;
        }
        var c = this.tryParseLeftAngleBracket();
        if (c) {
          s += c;
          continue;
        }
        break;
      }
      var d = U(n, this.clonePosition());
      return {
        val: { type: K.literal, value: s, location: d },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Uu(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return cn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), cn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(G.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(G.EMPTY_ARGUMENT, U(n, this.clonePosition()));
      var s = this.parseIdentifierIfPossible().value;
      if (!s)
        return this.error(G.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(G.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: K.argument,
              // value does not include the opening and closing braces.
              value: s,
              location: U(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(G.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition())) : this.parseArgumentOptions(t, r, s, n);
        default:
          return this.error(G.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = hn(this.message, r), s = r + n.length;
      this.bumpTo(s);
      var u = this.clonePosition(), l = U(t, u);
      return { value: n, location: l };
    }, e.prototype.parseArgumentOptions = function(t, r, n, s) {
      var u, l = this.clonePosition(), c = this.parseIdentifierIfPossible().value, d = this.clonePosition();
      switch (c) {
        case "":
          return this.error(G.EXPECT_ARGUMENT_TYPE, U(l, d));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var h = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var p = this.clonePosition(), x = this.parseSimpleArgStyleIfPossible();
            if (x.err)
              return x;
            var y = Gu(x.val);
            if (y.length === 0)
              return this.error(G.EXPECT_ARGUMENT_STYLE, U(this.clonePosition(), this.clonePosition()));
            var B = U(p, this.clonePosition());
            h = { style: y, styleLocation: B };
          }
          var H = this.tryParseArgumentClose(s);
          if (H.err)
            return H;
          var P = U(s, this.clonePosition());
          if (h && Si(h?.style, "::", 0)) {
            var O = Ru(h.style.slice(2));
            if (c === "number") {
              var x = this.parseNumberSkeletonFromString(O, h.styleLocation);
              return x.err ? x : {
                val: { type: K.number, value: n, location: P, style: x.val },
                err: null
              };
            } else {
              if (O.length === 0)
                return this.error(G.EXPECT_DATE_TIME_SKELETON, P);
              var E = O;
              this.locale && (E = Su(O, this.locale));
              var y = {
                type: Gt.dateTime,
                pattern: E,
                location: h.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? gu(E) : {}
              }, A = c === "date" ? K.date : K.time;
              return {
                val: { type: A, value: n, location: P, style: y },
                err: null
              };
            }
          }
          return {
            val: {
              type: c === "number" ? K.number : c === "date" ? K.date : K.time,
              value: n,
              location: P,
              style: (u = h?.style) !== null && u !== void 0 ? u : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var T = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(G.EXPECT_SELECT_ARGUMENT_OPTIONS, U(T, V({}, T)));
          this.bumpSpace();
          var g = this.parseIdentifierIfPossible(), b = 0;
          if (c !== "select" && g.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(G.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, U(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var x = this.tryParseDecimalInteger(G.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, G.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (x.err)
              return x;
            this.bumpSpace(), g = this.parseIdentifierIfPossible(), b = x.val;
          }
          var M = this.tryParsePluralOrSelectOptions(t, c, r, g);
          if (M.err)
            return M;
          var H = this.tryParseArgumentClose(s);
          if (H.err)
            return H;
          var C = U(s, this.clonePosition());
          return c === "select" ? {
            val: {
              type: K.select,
              value: n,
              options: Ti(M.val),
              location: C
            },
            err: null
          } : {
            val: {
              type: K.plural,
              value: n,
              options: Ti(M.val),
              offset: b,
              pluralType: c === "plural" ? "cardinal" : "ordinal",
              location: C
            },
            err: null
          };
        }
        default:
          return this.error(G.INVALID_ARGUMENT_TYPE, U(l, d));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(G.EXPECT_ARGUMENT_CLOSING_BRACE, U(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var s = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(G.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, U(s, this.clonePosition()));
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
        n = mu(t);
      } catch {
        return this.error(G.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: Gt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Eu(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, s) {
      for (var u, l = !1, c = [], d = /* @__PURE__ */ new Set(), h = s.value, p = s.location; ; ) {
        if (h.length === 0) {
          var x = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var y = this.tryParseDecimalInteger(G.EXPECT_PLURAL_ARGUMENT_SELECTOR, G.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (y.err)
              return y;
            p = U(x, this.clonePosition()), h = this.message.slice(x.offset, this.offset());
          } else
            break;
        }
        if (d.has(h))
          return this.error(r === "select" ? G.DUPLICATE_SELECT_ARGUMENT_SELECTOR : G.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, p);
        h === "other" && (l = !0), this.bumpSpace();
        var B = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? G.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : G.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, U(this.clonePosition(), this.clonePosition()));
        var H = this.parseMessage(t + 1, r, n);
        if (H.err)
          return H;
        var P = this.tryParseArgumentClose(B);
        if (P.err)
          return P;
        c.push([
          h,
          {
            value: H.val,
            location: U(B, this.clonePosition())
          }
        ]), d.add(h), this.bumpSpace(), u = this.parseIdentifierIfPossible(), h = u.value, p = u.location;
      }
      return c.length === 0 ? this.error(r === "select" ? G.EXPECT_SELECT_ARGUMENT_SELECTOR : G.EXPECT_PLURAL_ARGUMENT_SELECTOR, U(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !l ? this.error(G.MISSING_OTHER_CLAUSE, U(this.clonePosition(), this.clonePosition())) : { val: c, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, s = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var u = !1, l = 0; !this.isEOF(); ) {
        var c = this.char();
        if (c >= 48 && c <= 57)
          u = !0, l = l * 10 + (c - 48), this.bump();
        else
          break;
      }
      var d = U(s, this.clonePosition());
      return u ? (l *= n, Cu(l) ? { val: l, err: null } : this.error(r, d)) : this.error(t, d);
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
      var r = ga(this.message, t);
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
      if (Si(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && ma(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function dn(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Uu(e) {
  return dn(e) || e === 47;
}
function Fu(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function ma(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function ku(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function pn(e) {
  e.forEach(function(t) {
    if (delete t.location, ua(t) || la(t))
      for (var r in t.options)
        delete t.options[r].location, pn(t.options[r].value);
    else aa(t) && ca(t.style) || (sa(t) || oa(t)) && ln(t.style) ? delete t.style.location : fa(t) && pn(t.children);
  });
}
function ju(e, t) {
  t === void 0 && (t = {}), t = V({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Du(e, t).parse();
  if (r.err) {
    var n = SyntaxError(G[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || pn(r.val), r.val;
}
var Dt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Dt || (Dt = {}));
var Cr = (
  /** @class */
  (function(e) {
    Br(t, e);
    function t(r, n, s) {
      var u = e.call(this, r) || this;
      return u.code = n, u.originalMessage = s, u;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Ai = (
  /** @class */
  (function(e) {
    Br(t, e);
    function t(r, n, s, u) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(s).join('", "'), '"'), Dt.INVALID_VALUE, u) || this;
    }
    return t;
  })(Cr)
), Vu = (
  /** @class */
  (function(e) {
    Br(t, e);
    function t(r, n, s) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Dt.INVALID_VALUE, s) || this;
    }
    return t;
  })(Cr)
), Xu = (
  /** @class */
  (function(e) {
    Br(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Dt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Cr)
), ge;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ge || (ge = {}));
function Wu(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ge.literal || r.type !== ge.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function zu(e) {
  return typeof e == "function";
}
function Er(e, t, r, n, s, u, l) {
  if (e.length === 1 && _i(e[0]))
    return [
      {
        type: ge.literal,
        value: e[0].value
      }
    ];
  for (var c = [], d = 0, h = e; d < h.length; d++) {
    var p = h[d];
    if (_i(p)) {
      c.push({
        type: ge.literal,
        value: p.value
      });
      continue;
    }
    if (pu(p)) {
      typeof u == "number" && c.push({
        type: ge.literal,
        value: r.getNumberFormat(t).format(u)
      });
      continue;
    }
    var x = p.value;
    if (!(s && x in s))
      throw new Xu(x, l);
    var y = s[x];
    if (du(p)) {
      (!y || typeof y == "string" || typeof y == "number") && (y = typeof y == "string" || typeof y == "number" ? String(y) : ""), c.push({
        type: typeof y == "string" ? ge.literal : ge.object,
        value: y
      });
      continue;
    }
    if (sa(p)) {
      var B = typeof p.style == "string" ? n.date[p.style] : ln(p.style) ? p.style.parsedOptions : void 0;
      c.push({
        type: ge.literal,
        value: r.getDateTimeFormat(t, B).format(y)
      });
      continue;
    }
    if (oa(p)) {
      var B = typeof p.style == "string" ? n.time[p.style] : ln(p.style) ? p.style.parsedOptions : n.time.medium;
      c.push({
        type: ge.literal,
        value: r.getDateTimeFormat(t, B).format(y)
      });
      continue;
    }
    if (aa(p)) {
      var B = typeof p.style == "string" ? n.number[p.style] : ca(p.style) ? p.style.parsedOptions : void 0;
      B && B.scale && (y = y * (B.scale || 1)), c.push({
        type: ge.literal,
        value: r.getNumberFormat(t, B).format(y)
      });
      continue;
    }
    if (fa(p)) {
      var H = p.children, P = p.value, O = s[P];
      if (!zu(O))
        throw new Vu(P, "function", l);
      var E = Er(H, t, r, n, s, u), A = O(E.map(function(b) {
        return b.value;
      }));
      Array.isArray(A) || (A = [A]), c.push.apply(c, A.map(function(b) {
        return {
          type: typeof b == "string" ? ge.literal : ge.object,
          value: b
        };
      }));
    }
    if (ua(p)) {
      var T = p.options[y] || p.options.other;
      if (!T)
        throw new Ai(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Er(T.value, t, r, n, s));
      continue;
    }
    if (la(p)) {
      var T = p.options["=".concat(y)];
      if (!T) {
        if (!Intl.PluralRules)
          throw new Cr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Dt.MISSING_INTL_API, l);
        var g = r.getPluralRules(t, { type: p.pluralType }).select(y - (p.offset || 0));
        T = p.options[g] || p.options.other;
      }
      if (!T)
        throw new Ai(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Er(T.value, t, r, n, s, y - (p.offset || 0)));
      continue;
    }
  }
  return Wu(c);
}
function Yu(e, t) {
  return t ? V(V(V({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = V(V({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function qu(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Yu(e[n], t[n]), r;
  }, V({}, e)) : e;
}
function rn(e) {
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
function Zu(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: Qr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Kr([void 0], r, !1)))();
    }, {
      cache: rn(e.number),
      strategy: en.variadic
    }),
    getDateTimeFormat: Qr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Kr([void 0], r, !1)))();
    }, {
      cache: rn(e.dateTime),
      strategy: en.variadic
    }),
    getPluralRules: Qr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Kr([void 0], r, !1)))();
    }, {
      cache: rn(e.pluralRules),
      strategy: en.variadic
    })
  };
}
var Ju = (
  /** @class */
  (function() {
    function e(t, r, n, s) {
      r === void 0 && (r = e.defaultLocale);
      var u = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(d) {
        var h = u.formatToParts(d);
        if (h.length === 1)
          return h[0].value;
        var p = h.reduce(function(x, y) {
          return !x.length || y.type !== ge.literal || typeof x[x.length - 1] != "string" ? x.push(y.value) : x[x.length - 1] += y.value, x;
        }, []);
        return p.length <= 1 ? p[0] || "" : p;
      }, this.formatToParts = function(d) {
        return Er(u.ast, u.locales, u.formatters, u.formats, d, void 0, u.message);
      }, this.resolvedOptions = function() {
        var d;
        return {
          locale: ((d = u.resolvedLocale) === null || d === void 0 ? void 0 : d.toString()) || Intl.NumberFormat.supportedLocalesOf(u.locales)[0]
        };
      }, this.getAst = function() {
        return u.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var l = s || {};
        l.formatters;
        var c = au(l, ["formatters"]);
        this.ast = e.__parse(t, V(V({}, c), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = qu(e.formats, n), this.formatters = s && s.formatters || Zu(this.formatterCache);
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
    }, e.__parse = ju, e.formats = {
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
function $u(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let s = 0; s < r.length; s++)
    if (typeof n == "object") {
      if (s > 0) {
        const u = r.slice(s, r.length).join(".");
        if (u in n) {
          n = n[u];
          break;
        }
      }
      n = n[r[s]];
    } else
      n = void 0;
  return n;
}
const ct = {}, Ku = (e, t, r) => r && (t in ct || (ct[t] = {}), e in ct[t] || (ct[t][e] = r), r), _a = (e, t) => {
  if (t == null)
    return;
  if (t in ct && e in ct[t])
    return ct[t][e];
  const r = Lr(t);
  for (let n = 0; n < r.length; n++) {
    const s = r[n], u = el(s, e);
    if (u)
      return Ku(e, t, u);
  }
};
let Tn;
const ar = Pr({});
function Qu(e) {
  return Tn[e] || null;
}
function ya(e) {
  return e in Tn;
}
function el(e, t) {
  if (!ya(e))
    return null;
  const r = Qu(e);
  return $u(r, t);
}
function tl(e) {
  if (e == null)
    return;
  const t = Lr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (ya(n))
      return n;
  }
}
function rl(e, ...t) {
  delete ct[e], ar.update((r) => (r[e] = iu.all([r[e] || {}, ...t]), r));
}
Ft(
  [ar],
  ([e]) => Object.keys(e)
);
ar.subscribe((e) => Tn = e);
const Sr = {};
function nl(e, t) {
  Sr[e].delete(t), Sr[e].size === 0 && delete Sr[e];
}
function xa(e) {
  return Sr[e];
}
function il(e) {
  return Lr(e).map((t) => {
    const r = xa(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function vn(e) {
  return e == null ? !1 : Lr(e).some(
    (t) => {
      var r;
      return (r = xa(t)) == null ? void 0 : r.size;
    }
  );
}
function al(e, t) {
  return Promise.all(
    t.map((n) => (nl(e, n), n().then((s) => s.default || s)))
  ).then((n) => rl(e, ...n));
}
const Qt = {};
function Ea(e) {
  if (!vn(e))
    return e in Qt ? Qt[e] : Promise.resolve();
  const t = il(e);
  return Qt[e] = Promise.all(
    t.map(
      ([r, n]) => al(r, n)
    )
  ).then(() => {
    if (vn(e))
      return Ea(e);
    delete Qt[e];
  }), Qt[e];
}
const sl = {
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
}, ol = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: sl,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, ul = ol;
function Ut() {
  return ul;
}
const nn = Pr(!1);
var ll = Object.defineProperty, fl = Object.defineProperties, cl = Object.getOwnPropertyDescriptors, Mi = Object.getOwnPropertySymbols, hl = Object.prototype.hasOwnProperty, dl = Object.prototype.propertyIsEnumerable, Oi = (e, t, r) => t in e ? ll(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, pl = (e, t) => {
  for (var r in t || (t = {}))
    hl.call(t, r) && Oi(e, r, t[r]);
  if (Mi)
    for (var r of Mi(t))
      dl.call(t, r) && Oi(e, r, t[r]);
  return e;
}, vl = (e, t) => fl(e, cl(t));
let gn;
const Or = Pr(null);
function Hi(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Lr(e, t = Ut().fallbackLocale) {
  const r = Hi(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Hi(t)])] : r;
}
function Tt() {
  return gn ?? void 0;
}
Or.subscribe((e) => {
  gn = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const gl = (e) => {
  if (e && tl(e) && vn(e)) {
    const { loadingDelay: t } = Ut();
    let r;
    return typeof window < "u" && Tt() != null && t ? r = window.setTimeout(
      () => nn.set(!0),
      t
    ) : nn.set(!0), Ea(e).then(() => {
      Or.set(e);
    }).finally(() => {
      clearTimeout(r), nn.set(!1);
    });
  }
  return Or.set(e);
}, kt = vl(pl({}, Or), {
  set: gl
}), Rr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const s = JSON.stringify(n);
    return s in t ? t[s] : t[s] = e(n);
  };
};
var bl = Object.defineProperty, Hr = Object.getOwnPropertySymbols, Sa = Object.prototype.hasOwnProperty, Ta = Object.prototype.propertyIsEnumerable, Ii = (e, t, r) => t in e ? bl(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, wn = (e, t) => {
  for (var r in t || (t = {}))
    Sa.call(t, r) && Ii(e, r, t[r]);
  if (Hr)
    for (var r of Hr(t))
      Ta.call(t, r) && Ii(e, r, t[r]);
  return e;
}, jt = (e, t) => {
  var r = {};
  for (var n in e)
    Sa.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Hr)
    for (var n of Hr(e))
      t.indexOf(n) < 0 && Ta.call(e, n) && (r[n] = e[n]);
  return r;
};
const ir = (e, t) => {
  const { formats: r } = Ut();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, ml = Rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (s = ir("number", n)), new Intl.NumberFormat(r, s);
  }
), _l = Rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? s = ir("date", n) : Object.keys(s).length === 0 && (s = ir("date", "short")), new Intl.DateTimeFormat(r, s);
  }
), yl = Rr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? s = ir("time", n) : Object.keys(s).length === 0 && (s = ir("time", "short")), new Intl.DateTimeFormat(r, s);
  }
), xl = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return ml(wn({ locale: r }, n));
}, El = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return _l(wn({ locale: r }, n));
}, Sl = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return yl(wn({ locale: r }, n));
}, Tl = Rr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Tt()) => new Ju(e, t, Ut().formats, {
    ignoreTag: Ut().ignoreTag
  })
), wl = (e, t = {}) => {
  var r, n, s, u;
  let l = t;
  typeof e == "object" && (l = e, e = l.id);
  const {
    values: c,
    locale: d = Tt(),
    default: h
  } = l;
  if (d == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let p = _a(e, d);
  if (!p)
    p = (u = (s = (n = (r = Ut()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: d, id: e, defaultValue: h })) != null ? s : h) != null ? u : e;
  else if (typeof p != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof p}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), p;
  if (!c)
    return p;
  let x = p;
  try {
    x = Tl(p, d).format(c);
  } catch (y) {
    y instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      y.message
    );
  }
  return x;
}, Al = (e, t) => Sl(t).format(e), Ml = (e, t) => El(t).format(e), Ol = (e, t) => xl(t).format(e), Hl = (e, t = Tt()) => _a(e, t);
Ft([kt, ar], () => wl);
Ft([kt], () => Al);
Ft([kt], () => Ml);
Ft([kt], () => Ol);
Ft([kt, ar], () => Hl);
const Il = "__i18n__", Nl = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Pl = [
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
function Bl(e) {
  return typeof e == "string" && e.includes(Il);
}
class Cl {
  load_component;
  #t = Z(ft({}));
  get shared() {
    return a(this.#t);
  }
  set shared(t) {
    _(this.#t, t, !0);
  }
  #r = Z(ft({}));
  get props() {
    return a(this.#r);
  }
  set props(t) {
    _(this.#r, t, !0);
  }
  #e = Z((t) => t);
  get i18n() {
    return a(this.#e);
  }
  set i18n(t) {
    _(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Pl;
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
    for (const n of Nl)
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
    ), Rt(() => {
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
    }), Object.keys(this.translatable_props).length > 0 && kt.subscribe(() => {
      for (const [n, s] of Object.entries(this.translatable_props)) {
        const [u, l] = n.split("."), c = this.i18n(s);
        u === "shared" ? this.shared[l] = c : this.props[l] = c;
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
    const s = this.i18n(n);
    return s !== n && (this.translatable_props[`${t}.${r}`] = n), s;
  }
  dispatch(t, r) {
    this.dispatcher(this.shared.id, t, r);
  }
  async get_data() {
    return yo(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], s = Bl(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const u = r;
        this.shared[u] = s;
        continue;
      }
      this.props[r] = s;
    }
  }
  watch_for_change() {
    Rt(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
go();
var Ll = /* @__PURE__ */ Oo('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Ni = /* @__PURE__ */ xe("<!> <!>", 1), Rl = /* @__PURE__ */ xe('<div class="placeholder svelte-1stq1b1"></div>');
function Gl(e, t) {
  Yi(t, !1);
  let r = te(t, "height", 8, void 0), n = te(t, "min_height", 8, void 0), s = te(t, "max_height", 8, void 0), u = te(t, "width", 8, void 0), l = te(t, "elem_id", 8, ""), c = te(t, "elem_classes", 24, () => []), d = te(t, "variant", 8, "solid"), h = te(t, "border_mode", 8, "base"), p = te(t, "padding", 8, !0), x = te(t, "type", 8, "normal"), y = te(t, "test_id", 8, void 0), B = te(t, "explicit_call", 8, !1), H = te(t, "container", 8, !0), P = te(t, "visible", 8, !0), O = te(t, "allow_overflow", 8, !0), E = te(t, "overflow_behavior", 8, "auto"), A = te(t, "scale", 8, null), T = te(t, "min_width", 8, 0), g = te(t, "flex", 12, !1), b = te(t, "resizable", 8, !1), M = te(t, "rtl", 8, !1), C = te(t, "fullscreen", 12, !1), D = te(t, "label", 8, void 0), re = Pt(C()), k = Pt(), Le = x() === "fieldset" ? "fieldset" : "div", Re = Pt(0), wt = Pt(0), Q = Pt(null);
  function ht(ie) {
    C() && ie.key === "Escape" && C(!1);
  }
  const ce = (ie) => {
    if (ie !== void 0) {
      if (typeof ie == "number")
        return ie + "px";
      if (typeof ie == "string")
        return ie;
    }
  }, dt = (ie) => {
    let De = ie.clientY;
    const Ye = (Be) => {
      const de = Be.clientY - De;
      De = Be.clientY, mo(k, a(k).style.height = `${a(k).offsetHeight + de}px`);
    }, qe = () => {
      window.removeEventListener("mousemove", Ye), window.removeEventListener("mouseup", qe);
    };
    window.addEventListener("mousemove", Ye), window.addEventListener("mouseup", qe);
  };
  li(
    () => (Ne(C()), a(re), a(k)),
    () => {
      C() !== a(re) && (_(re, C()), C() ? (_(Q, a(k).getBoundingClientRect()), _(Re, a(k).offsetHeight), _(wt, a(k).offsetWidth), window.addEventListener("keydown", ht)) : (_(Q, null), window.removeEventListener("keydown", ht)));
    }
  ), li(() => Ne(P()), () => {
    P() || g(!1);
  }), bo(), $o();
  var Pe = Ho(), Ge = Yr(Pe);
  {
    var Qe = (ie) => {
      var De = Ni(), Ye = Yr(De);
      Go(Ye, () => Le, !1, (de, Ue) => {
        ra(de, (ee) => _(k, ee), () => a(k)), Jo(
          de,
          (ee, tt) => ({
            "data-testid": y(),
            id: l(),
            class: `block ${ee ?? ""}`,
            dir: M() ? "rtl" : "ltr",
            "aria-label": D(),
            style: "",
            [tr]: {
              hidden: P() === "hidden",
              padded: p(),
              flex: g(),
              border_focus: h() === "focus",
              border_contrast: h() === "contrast",
              "hide-container": !B() && !H(),
              fullscreen: C(),
              animating: C() && a(Q) !== null,
              "auto-margin": A() === null
            },
            [Bt]: tt
          }),
          [
            () => (Ne(c()), he(() => c()?.join(" ") || "")),
            () => ({
              height: (Ne(C()), Ne(r()), he(() => C() ? void 0 : ce(r()))),
              "min-height": (Ne(C()), Ne(n()), he(() => C() ? void 0 : ce(n()))),
              "max-height": (Ne(C()), Ne(s()), he(() => C() ? void 0 : ce(s()))),
              "--start-top": (a(Q), he(() => a(Q) ? `${a(Q).top}px` : "0px")),
              "--start-left": (a(Q), he(() => a(Q) ? `${a(Q).left}px` : "0px")),
              "--start-width": (a(Q), he(() => a(Q) ? `${a(Q).width}px` : "0px")),
              "--start-height": (a(Q), he(() => a(Q) ? `${a(Q).height}px` : "0px")),
              width: (Ne(C()), Ne(u()), he(() => C() ? void 0 : typeof u() == "number" ? `calc(min(${u()}px, 100%))` : ce(u()))),
              "border-style": d(),
              overflow: O() ? E() : "hidden",
              "flex-grow": A(),
              "min-width": `calc(min(${T()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Me = Ni(), At = Yr(Me);
        Ro(At, t, "default", {});
        var Vt = z(At, 2);
        {
          var et = (ee) => {
            var tt = Ll();
            j("mousedown", tt, dt), fe(ee, tt);
          };
          We(Vt, (ee) => {
            b() && ee(et);
          });
        }
        fe(Ue, Me);
      });
      var qe = z(Ye, 2);
      {
        var Be = (de) => {
          var Ue = Rl();
          let Me;
          Ke(() => Me = Ar(Ue, "", Me, {
            height: a(Re) + "px",
            width: a(wt) + "px"
          })), fe(de, Ue);
        };
        We(qe, (de) => {
          C() && de(Be);
        });
      }
      fe(ie, De);
    };
    We(Ge, (ie) => {
      (P() === !0 || P() === "hidden") && ie(Qe);
    });
  }
  fe(e, Pe), zi();
}
En(["click"]);
En(["click"]);
var Dl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), Ul = /* @__PURE__ */ xe("<option> </option>"), Fl = /* @__PURE__ */ xe('<span class="current-badge svelte-r41nsf">当前</span>'), kl = /* @__PURE__ */ xe('<div role="button" tabindex="0"><span class="group-drag-handle svelte-r41nsf" title="拖动调整叠层顺序" aria-label="拖动调整叠层顺序">⠿</span> <span class="group-order svelte-r41nsf"></span> <span class="group-reorder-label svelte-r41nsf"> <!></span> <button type="button" class="visibility-button svelte-r41nsf"> </button> <button type="button" title="只显示此层">S</button></div>'), jl = /* @__PURE__ */ xe('<div class="group-reorder svelte-r41nsf" aria-label="调整 Label 叠层顺序"><div class="group-reorder-hint svelte-r41nsf">拖动排序；眼睛与单层显示仅影响浏览器预览</div> <div class="group-reorder-list svelte-r41nsf"></div></div>'), Vl = /* @__PURE__ */ xe('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select> <div class="mode-switch svelte-r41nsf" aria-label="显示模式"><button type="button">对齐模式</button> <button type="button">交叠预览</button></div> <!></div>'), Xl = /* @__PURE__ */ xe('<label class="svelte-r41nsf"><span> </span> <input type="number" class="svelte-r41nsf"/> <small class="svelte-r41nsf"> </small></label>'), Wl = /* @__PURE__ */ xe('<div class="numeric-toolbar svelte-r41nsf" aria-label="当前层精细参数"></div>'), zl = /* @__PURE__ */ xe('<button type="button" class="svelte-r41nsf"> </button>'), Yl = /* @__PURE__ */ xe('<div class="proposal-actions svelte-r41nsf"></div>'), ql = /* @__PURE__ */ xe('<div><div class="proposal-copy svelte-r41nsf"><strong class="svelte-r41nsf"> </strong> <span> </span></div> <!></div>'), Zl = /* @__PURE__ */ xe('<div class="overlap-legend svelte-r41nsf"><span class="legend-current svelte-r41nsf"> </span> <span class="legend-next svelte-r41nsf"> </span> <span class="legend-overlap svelte-r41nsf"> </span></div>'), Jl = /* @__PURE__ */ xe('<div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回标注</button> <button type="button" class="svelte-r41nsf">画布 −</button> <button type="button" class="svelte-r41nsf"> </button> <button type="button" class="svelte-r41nsf">画布 ＋</button></div> <!> <!> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <!> <div class="status svelte-r41nsf"><span> </span> <span class="svelte-r41nsf"> </span></div></div>');
function Kl(e, t) {
  Yi(t, !0);
  const r = /* @__PURE__ */ Qo(t, Dl), n = 2048, s = Math.log(1.01) / 100, u = 250, l = 1e4, c = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], d = new Cl(r);
  let h, p = null, x = null, y = null, B = null, H = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), O = null, E = Z(!1), A = Z(!1), T = Z("等待版图 mask"), g = Z(ft({ enabled: !1 })), b = Z(ft(An())), M = Z(""), C = Z(!1), D = Z(!1), re = Z("crosshair"), k = "", Le = 0, Re = { x: 0, y: 0, center_x: 0, center_y: 0 }, wt = { angle: 0, rotation: 0 }, Q = null, ht = null, ce = null, dt = null, Pe = 0, Ge = 0, Qe = "", ie = "", De = "", Ye = /* @__PURE__ */ new Set(), qe = Z(""), Be = Z(""), de = Z("before"), Ue = Z(!1), Me = Z("align"), At = Z(ft(/* @__PURE__ */ new Set())), Vt = Z(ft(/* @__PURE__ */ new Set())), et = Z(""), ee = Z(null), tt = /* @__PURE__ */ new Set(), X = Z(1), Fe = Z(0), ke = Z(0), Mt = Z(!1), sr = Z(!1), Xt = { client_x: 0, client_y: 0, pan_x: 0, pan_y: 0 }, Wt = null, or = Z(""), rt = Z(ft({ x: "0", y: "0", scale: "100", rotation: "0", alpha: "22" }));
  function An() {
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
  function Mn(i) {
    return JSON.parse(JSON.stringify(i || { enabled: !1 }));
  }
  function Ee(i) {
    const o = Object.assign(Object.assign({}, An()), i || {});
    return Object.assign(Object.assign({}, o), {
      center_x: ae(o.center_x, 0),
      center_y: ae(o.center_y, 0),
      pivot_x: ae(o.pivot_x, 0),
      pivot_y: ae(o.pivot_y, 0),
      scale: Se(ae(o.scale, 1), 0.01, 20),
      rotation_deg: ur(ae(o.rotation_deg, 0)),
      preview_alpha: Se(ae(o.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(ae(o.revision, 0)))
    });
  }
  function ae(i, o) {
    const f = Number(i);
    return Number.isFinite(f) ? f : o;
  }
  function wa(i) {
    return typeof i == "number" ? String(i) + "px" : i || "520px";
  }
  function Se(i, o, f) {
    return Math.max(o, Math.min(f, i));
  }
  function ur(i) {
    let o = ((i + 180) % 360 + 360) % 360 - 180;
    return o === -180 && (o = 180), o;
  }
  function F(i) {
    return typeof i + ":" + String(i);
  }
  function je() {
    var i;
    return Y() ? Array.isArray((i = a(g).group_view) === null || i === void 0 ? void 0 : i.groups) ? a(g).group_view.groups : [] : [];
  }
  function Ot() {
    var i;
    return Y() && ((i = a(g).group_view) === null || i === void 0 ? void 0 : i.allow_reorder) === !0;
  }
  function Y() {
    return a(g).transform_mode === "label_groups" && !!a(g).group_view;
  }
  function On() {
    for (const i of je())
      if (F(i.group_id) === a(M)) return i;
    return null;
  }
  function pe() {
    const i = On();
    return i ? String(i.label || "Label " + String(i.group_id)) : "";
  }
  function zt() {
    const i = je(), o = i.findIndex((f) => F(f.group_id) === a(M));
    return o < 0 || o + 1 >= i.length ? "" : F(i[o + 1].group_id);
  }
  function Aa(i) {
    return a(et) ? i === a(et) : i === a(M) ? !0 : a(Me) === "overlap" && i === zt();
  }
  function lr(i) {
    return a(At).has(i) ? !1 : a(Vt).has(i) ? !0 : Aa(i);
  }
  function Hn() {
    return je().map((i) => F(i.group_id)).filter((i) => lr(i));
  }
  function ve() {
    return Math.max(1, Number(a(g).target_width || p?.naturalWidth || 1));
  }
  function Te() {
    return Math.max(1, Number(a(g).target_height || p?.naturalHeight || 1));
  }
  function Ma() {
    var i;
    const o = (i = H.get(a(M))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(g).source_width || x?.naturalWidth || o?.naturalWidth || 1));
  }
  function Oa() {
    var i;
    const o = (i = H.get(a(M))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(g).source_height || x?.naturalHeight || o?.naturalHeight || 1));
  }
  function Ha() {
    return Math.min(1, n / Math.max(ve(), Te()));
  }
  function In(i = a(M)) {
    var o;
    if (Y()) {
      const v = (o = H.get(i)) === null || o === void 0 ? void 0 : o.view.foreground_bbox_xyxy;
      if (Array.isArray(v) && v.length >= 4) return v.slice(0, 4).map(Number);
    }
    const f = a(g).foreground_bbox_xyxy;
    return Array.isArray(f) && f.length >= 4 ? f.slice(0, 4).map(Number) : [0, 0, Ma() - 1, Oa() - 1];
  }
  function Gr(i, o, f) {
    if (!i) {
      o === Le && f(null);
      return;
    }
    const v = new Image();
    v.onload = () => {
      o === Le && f(v);
    }, v.onerror = () => {
      o === Le && f(null);
    }, v.src = i;
  }
  function Nn(i, o) {
    const f = Math.max(1, Number(a(g).source_width || i.naturalWidth || 1)), v = Math.max(1, Number(a(g).source_height || i.naturalHeight || 1)), m = document.createElement("canvas");
    m.width = f, m.height = v;
    const S = m.getContext("2d", { willReadFrequently: !0 });
    if (!S) return null;
    S.imageSmoothingEnabled = !1, S.drawImage(i, 0, 0, f, v);
    const w = S.getImageData(0, 0, f, v), I = document.createElement("canvas");
    I.width = f, I.height = v;
    const N = I.getContext("2d");
    if (!N) return null;
    const L = N.createImageData(f, v);
    for (let R = 0; R < w.data.length; R += 4) {
      const J = Math.max(w.data[R], w.data[R + 1], w.data[R + 2]);
      w.data[R + 3] > 0 && J >= 128 && (L.data[R] = o[0], L.data[R + 1] = o[1], L.data[R + 2] = o[2], L.data[R + 3] = 255);
    }
    return N.putImageData(L, 0, 0), { mask: m, tint: I };
  }
  function Ia() {
    if (!x) {
      y = null, B = null;
      return;
    }
    const i = Nn(x, c[0]);
    y = i?.mask || null, B = i?.tint || null;
  }
  function Na(i) {
    if (!i.image) {
      i.maskCanvas = null, i.tintCanvas = null, i.ready = !1;
      return;
    }
    const o = Nn(i.image, i.color);
    i.maskCanvas = o?.mask || null, i.tintCanvas = o?.tint || null, i.ready = !!o;
  }
  function Ht() {
    Q && (clearTimeout(Q), Q = null);
  }
  function Yt() {
    ht && (clearTimeout(ht), ht = null);
  }
  function Pa(i) {
    const o = JSON.stringify(i);
    let f = 2166136261;
    for (let v = 0; v < o.length; v++)
      f ^= o.charCodeAt(v), f = Math.imul(f, 16777619);
    return "fnv1a32:" + (f >>> 0).toString(16).padStart(8, "0");
  }
  function be() {
    Wt === null && (Wt = requestAnimationFrame(() => {
      Wt = null, vt();
    }));
  }
  function It() {
    a(or) || _(
      rt,
      {
        x: ae(a(b).center_x, 0).toFixed(2).replace(/\.00$/, ""),
        y: ae(a(b).center_y, 0).toFixed(2).replace(/\.00$/, ""),
        scale: (ae(a(b).scale, 1) * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        rotation: ae(a(b).rotation_deg, 0).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        alpha: (ae(a(b).preview_alpha, 0.22) * 100).toFixed(1).replace(/\.0$/, "")
      },
      !0
    );
  }
  function Ba(i, o) {
    if (!Array.isArray(o) || o.length !== i.length) return i;
    const f = /* @__PURE__ */ new Map();
    for (const S of i) {
      if (!S || S.group_id === void 0 || S.group_id === null) return i;
      const w = F(S.group_id);
      if (f.has(w)) return i;
      f.set(w, S);
    }
    const v = [], m = /* @__PURE__ */ new Set();
    for (const S of o) {
      if (typeof S != "string" && typeof S != "number") return i;
      const w = F(S), I = f.get(w);
      if (!I || m.has(w)) return i;
      m.add(w), v.push(I);
    }
    return v.length === i.length ? v : i;
  }
  function Ca(i) {
    var o;
    const f = a(g).group_view, v = Array.isArray(f?.groups) ? f.groups : [], m = a(g).group_intent, S = !!f && !!m && String(m.selection_signature || "") === String(f.selection_signature || "");
    Qe = String(a(g).editor_session_id || m?.editor_session_id || ""), ie = String(a(g).document_epoch || m?.document_epoch || f?.selection_signature || ""), Pe = Math.max(0, Math.trunc(ae(
      (o = m?.client_revision) !== null && o !== void 0 ? o : m?.transform_set_revision,
      0
    ))), Ge = Pe, ce = null, Yt();
    const w = f?.allow_reorder === !0 && S ? Ba(v, m?.ordered_group_ids) : v;
    w !== v && f && _(
      g,
      Object.assign(Object.assign({}, a(g)), {
        group_view: Object.assign(Object.assign({}, f), { groups: w })
      }),
      !0
    );
    const I = /* @__PURE__ */ new Map();
    if (S && Array.isArray(m?.transforms))
      for (const q of m.transforms)
        !q || q.group_id === void 0 || !q.transform || I.set(F(q.group_id), Ee(q.transform));
    const N = Ee(a(g).transform), L = /* @__PURE__ */ new Map(), R = /* @__PURE__ */ new Map(), J = /* @__PURE__ */ new Set();
    for (let q = 0; q < w.length; q++) {
      const Oe = w[q];
      if (!Oe || Oe.group_id === void 0 || Oe.group_id === null) continue;
      const Ze = F(Oe.group_id);
      if (J.has(Ze)) continue;
      J.add(Ze);
      const Je = {
        view: Oe,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: c[q % c.length]
      };
      L.set(Ze, Je), R.set(Ze, Ee(I.get(Ze) || N));
    }
    H = L, P = R, Ye = /* @__PURE__ */ new Set(), tt = /* @__PURE__ */ new Set(), _(ee, null);
    const W = S && m?.active_group_id !== void 0 && m.active_group_id !== null ? F(m.active_group_id) : "";
    _(
      M,
      L.has(W) ? W : L.keys().next().value || "",
      !0
    ), _(b, Ee(P.get(a(M)) || N), !0), a(M) && P.set(a(M), a(b)), It(), x = null, y = null, B = null, _(A, !1);
    for (const [q, Oe] of L)
      Gr(Oe.view.mask_image, i, (Ze) => {
        const Je = H.get(q);
        !Je || Je !== Oe || (Je.image = Ze, Na(Je), vt());
      });
    _(
      T,
      a(g).status || (w.length > 0 ? "已加载 " + String(w.length) + " 个 Label；当前：" + pe() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function La(i) {
    Ht(), Yt(), ce = null, dt = null, Le += 1;
    const o = Le;
    _(g, Mn(i), !0), _(E, !1), H = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), _(M, ""), _(C, !1), _(D, !1), Gr(a(g).base_image, o, (f) => {
      p = f, _(E, !!f), vt();
    }), Y() ? Ca(o) : (_(b, Ee(a(g).transform), !0), It(), _(T, a(g).status || "编辑器已加载", !0), _(A, !1), Gr(a(g).mask_image, o, (f) => {
      x = f, _(A, !!f), Ia(), vt();
    }));
  }
  function Pn() {
    return Array.isArray(a(ee) === null || a(ee) === void 0 ? void 0 : a(ee).candidates) ? a(ee).candidates : [];
  }
  function Bn(i, o = !1) {
    if (!a(ee) || a(ee).target_layer_id === void 0) return;
    const f = F(a(ee).target_layer_id);
    if (!P.has(f)) return;
    if (o && (tt.has(f) || ae(Ve(f).revision, 0) !== ae(a(ee).start_transform_revision, 0))) {
      _(T, "粗对齐结果已到达，但检测到手动修改，未覆盖当前变换"), nt();
      return;
    }
    Jt(f);
    const v = Ee(Ve(f));
    Xe(Object.assign(Object.assign(Object.assign({}, v), i.transform), { preview_alpha: v.preview_alpha }), !1), _(ee, null), dr("boundary_alignment", (o ? "已自动应用 " : "已应用 ") + String(i.candidate_id || "候选") + "；停止 250ms 后同步");
  }
  function Ra(i) {
    const o = String(i.editor_session_id || ""), f = String(i.document_epoch || "");
    if (o && o !== Qe || f && f !== ie) return;
    _(ee, Mn(i), !0), _(T, String(i.status || "粗对齐候选已生成"), !0), nt();
    const v = Array.isArray(i.candidates) ? i.candidates : [];
    i.auto_apply && v.length ? Bn(v[0], !0) : be();
  }
  function Ga(i) {
    const o = String(i.editor_session_id || ""), f = String(i.document_epoch || "");
    if (o && o !== Qe || f && f !== ie)
      return;
    if (Yt(), i.payload_kind === "ack_error") {
      ce = null, _(T, String(i.status || "同步失败；本地修改仍保留"), !0), nt(), vt();
      return;
    }
    const v = Math.max(0, Math.trunc(ae(i.server_ack_revision, 0)));
    Ge = Math.max(Ge, v), dt = null, ce !== null && v >= ce && (ce = null), _(T, Pe > Ge ? "本地修改待同步" : "已同步", !0), nt(), Pe > Ge && cr(0), vt();
  }
  Rt(() => {
    const i = d.props.value || null, o = JSON.stringify(i);
    if (i?.payload_kind === "ack" || i?.payload_kind === "ack_error") {
      o !== De && (De = o, Ga(i));
      return;
    }
    if (i?.payload_kind === "alignment_proposal") {
      o !== De && (De = o, Ra(i));
      return;
    }
    o !== k && (k = o, La(i));
  }), No(() => {
    Le += 1, Ht(), Yt(), Wt !== null && cancelAnimationFrame(Wt);
  });
  function Dr(i) {
    const o = Number(i.rotation_deg || 0) * Math.PI / 180, f = Number(i.scale || 1), v = Math.cos(o), m = Math.sin(o), S = f * v, w = f * m, I = Number(i.center_x || 0) - S * Number(i.pivot_x || 0) + w * Number(i.pivot_y || 0), N = Number(i.center_y || 0) - w * Number(i.pivot_x || 0) - S * Number(i.pivot_y || 0);
    return [S, w, -w, S, I, N];
  }
  function qt(i, o, f = a(b)) {
    const [v, m, S, w, I, N] = Dr(f);
    return { x: v * i + S * o + I, y: m * i + w * o + N };
  }
  function Ur(i, o, f = a(b)) {
    const [v, m, S, w, I, N] = Dr(f), L = v * w - m * S;
    if (Math.abs(L) < 1e-9) return { x: -1, y: -1 };
    const R = i - I, J = o - N;
    return { x: (w * R - S * J) / L, y: (-m * R + v * J) / L };
  }
  function Zt(i) {
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * ve(), v = (i.clientY - o.top) / Math.max(1, o.height) * Te();
    return {
      x: (f - a(Fe)) / Math.max(0.05, a(X)),
      y: (v - a(ke)) / Math.max(0.05, a(X))
    };
  }
  function Fr(i, o) {
    i.setTransform(o * a(X), 0, 0, o * a(X), o * a(Fe), o * a(ke));
  }
  function Cn(i, o, f) {
    if (!i) return !1;
    const v = Math.round(o), m = Math.round(f);
    if (v < 0 || m < 0 || v >= i.width || m >= i.height) return !1;
    const S = i.getContext("2d", { willReadFrequently: !0 });
    if (!S) return !1;
    const w = S.getImageData(v, m, 1, 1).data;
    return w[3] > 0 && Math.max(w[0], w[1], w[2]) >= 128;
  }
  function Ve(i) {
    return !Y() || i === a(M) ? a(b) : P.get(i) || a(b);
  }
  function Xe(i, o = !0) {
    _(b, Ee(i), !0), Y() && a(M) && (P.set(a(M), a(b)), o && tt.add(a(M))), It();
  }
  function Ln(i, o) {
    const f = Ur(i, o, a(b));
    return Cn(y, f.x, f.y);
  }
  function Rn(i, o, f) {
    const v = H.get(i);
    if (!v?.ready) return !1;
    const m = Ur(o, f, Ve(i));
    return Cn(v.maskCanvas, m.x, m.y);
  }
  function kr(i, o) {
    const f = new Set(Hn());
    if (a(M) && f.has(a(M)) && Rn(a(M), i, o)) return a(M);
    const v = je().map((m) => F(m.group_id)).reverse();
    for (const m of v)
      if (m !== a(M) && f.has(m) && Rn(m, i, o)) return m;
    return "";
  }
  function fr(i = a(M), o = Ve(i)) {
    const [f, v, m, S] = In(i);
    return [
      qt(f, v, o),
      qt(m, v, o),
      qt(m, S, o),
      qt(f, S, o)
    ];
  }
  function Gn() {
    const i = fr();
    let o = i[0];
    for (const w of i)
      (w.y < o.y || Math.abs(w.y - o.y) < 1e-6 && w.x > o.x) && (o = w);
    const f = Number(a(b).rotation_deg || 0) * Math.PI / 180, v = Math.cos(f - Math.PI / 4), m = Math.sin(f - Math.PI / 4), S = 34 / Math.max(0.05, a(X));
    return { x: o.x + v * S, y: o.y + m * S };
  }
  function Dn() {
    const i = h?.getBoundingClientRect();
    return i ? Math.max(8 / a(X), 14 * ve() / Math.max(1, i.width) / a(X)) : 14 / a(X);
  }
  function Un(i, o) {
    if (Y() && !a(M)) return !1;
    const f = Gn(), v = Dn();
    return Math.hypot(i - f.x, o - f.y) <= v;
  }
  function Fn(i) {
    const o = Y() ? a(M) : "";
    Xe(Object.assign(Object.assign({}, a(b)), {
      revision: Number(a(b).revision || 0) + 1,
      origin: i,
      scale: Se(Number(a(b).scale || 1), 0.01, 20),
      rotation_deg: ur(Number(a(b).rotation_deg || 0))
    })), o && Ye.add(o);
  }
  function Da() {
    const i = [];
    for (const o of je()) {
      const f = F(o.group_id);
      i.push({
        group_id: o.group_id,
        transform: Object.assign({}, Ee(Ve(f)))
      });
    }
    return i;
  }
  function nt() {
    var i;
    const o = {
      payload_kind: "intent",
      editor_session_id: Qe,
      document_epoch: ie,
      payload_hash: String(((i = a(g).group_intent) === null || i === void 0 ? void 0 : i.payload_hash) || ""),
      enabled: a(g).enabled,
      transform: Object.assign({}, Ee(a(b))),
      target_width: a(g).target_width,
      target_height: a(g).target_height
    };
    Y() && (o.transform_mode = "label_groups", o.group_intent = a(g).group_intent ? JSON.parse(JSON.stringify(a(g).group_intent)) : null), d.props.value = o, k = JSON.stringify(o);
  }
  function Ua() {
    if (!Y()) {
      d.dispatch("change");
      return;
    }
    const i = a(g).group_intent, o = Math.max(0, Math.trunc(ae(i?.client_revision, 0)));
    !i || o <= Ge || ce !== null || (Yt(), ce = o, _(T, "同步中…"), nt(), d.dispatch("change"), ht = setTimeout(
      () => {
        if (ce !== o) return;
        ce = null;
        const f = dt !== o;
        dt = o, _(T, f ? "同步超时，正在重试；本地修改仍保留" : "同步失败；可继续编辑或直接保存重试", !0), nt(), vt(), f && cr(1e3);
      },
      l
    ));
  }
  function cr(i = u) {
    Ht(), Q = setTimeout(
      () => {
        Q = null, Ua();
      },
      Math.max(0, i)
    );
  }
  function hr(i, o, f, v = !0) {
    var m;
    if (!Y()) return;
    f && Fn(i);
    const S = a(g).group_view, w = On(), I = je().filter((R) => Ye.has(F(R.group_id))).map((R) => R.group_id), N = Pe + 1, L = Object.assign(
      Object.assign(
        {
          selection_signature: String(S?.selection_signature || ie),
          transform_set_revision: N,
          editor_session_id: Qe,
          document_epoch: ie,
          client_revision: N,
          payload_hash: "",
          active_group_id: (m = w?.group_id) !== null && m !== void 0 ? m : null,
          changed_group_ids: I
        },
        Ot() ? {
          ordered_group_ids: je().map((R) => R.group_id)
        } : {}
      ),
      { transforms: Da() }
    );
    L.payload_hash = Pa(Object.assign(Object.assign({}, L), { payload_hash: "" })), Pe = N, dt = null, _(
      g,
      Object.assign(Object.assign({}, a(g)), {
        payload_kind: "intent",
        editor_session_id: Qe,
        document_epoch: ie,
        payload_hash: L.payload_hash,
        group_intent: L
      }),
      !0
    ), _(T, o + "（本地修改）"), nt(), v && cr(0), be();
  }
  function pt(i, o, f = !0) {
    if (Y()) {
      hr(i, o, !0, f);
      return;
    }
    Fn(i), _(
      g,
      Object.assign(Object.assign({}, a(g)), {
        enabled: !0,
        transform: Object.assign({}, a(b)),
        status: o
      }),
      !0
    ), _(T, o, !0), nt(), f && (Ht(), d.dispatch("change")), be();
  }
  function dr(i, o, f = u) {
    if (Y()) {
      pt(i, o, !1), cr(f);
      return;
    }
    pt(i, o, !1), Ht(), Q = setTimeout(
      () => {
        Q = null, d.dispatch("change");
      },
      f
    );
  }
  function Fa(i, o, f) {
    if (p && a(E)) {
      i.imageSmoothingEnabled = !0, i.drawImage(p, 0, 0, o, f);
      return;
    }
    i.fillStyle = "#f8fafc", i.fillRect(0, 0, o, f), i.fillStyle = "#64748b", i.font = "18px sans-serif", i.fillText("请先上传图像", 24, 42);
  }
  function pr(i, o, f, v, m) {
    var S;
    m === void 0 && (m = Se(Number((S = v.preview_alpha) !== null && S !== void 0 ? S : 0.35), 0, 1)), i.save(), i.globalAlpha = m;
    const [w, I, N, L, R, J] = Dr(v);
    i.setTransform(o * a(X) * w, o * a(X) * I, o * a(X) * N, o * a(X) * L, o * (a(X) * R + a(Fe)), o * (a(X) * J + a(ke))), i.imageSmoothingEnabled = !1, i.drawImage(f, 0, 0, f.width, f.height), i.restore();
  }
  function ka(i, o, f, v) {
    const m = H.get(f), S = H.get(v);
    if (!m?.tintCanvas || !S?.tintCanvas || !m.ready || !S.ready) return;
    const w = O || document.createElement("canvas");
    O = w, w.width !== h.width && (w.width = h.width), w.height !== h.height && (w.height = h.height);
    const I = w.getContext("2d");
    if (I) {
      I.setTransform(1, 0, 0, 1, 0, 0), I.globalCompositeOperation = "source-over", I.clearRect(0, 0, w.width, w.height), pr(I, o, m.tintCanvas, Ve(f), 1), I.globalCompositeOperation = "destination-in", pr(I, o, S.tintCanvas, Ve(v), 1), I.globalCompositeOperation = "source-in", I.setTransform(1, 0, 0, 1, 0, 0), I.fillStyle = "#ff3fa4", I.fillRect(0, 0, w.width, w.height), I.globalCompositeOperation = "source-atop", I.strokeStyle = "rgba(255,255,255,0.78)", I.lineWidth = 1.5;
      for (let N = -w.height; N < w.width; N += 14)
        I.beginPath(), I.moveTo(N, 0), I.lineTo(N + w.height, w.height), I.stroke();
      i.save(), i.setTransform(1, 0, 0, 1, 0, 0), i.globalAlpha = 0.62, i.drawImage(w, 0, 0), i.restore();
    }
  }
  function kn(i, o) {
    const f = fr();
    i.save(), i.lineJoin = "round", i.lineWidth = Math.max(2.5, o / 700) / a(X), i.strokeStyle = "rgba(0,0,0,0.82)", i.beginPath(), i.moveTo(f[0].x, f[0].y);
    for (let S = 1; S < f.length; S++) i.lineTo(f[S].x, f[S].y);
    i.closePath(), i.stroke(), i.lineWidth = Math.max(1.8, o / 1e3) / a(X), i.strokeStyle = "#00ff66", i.stroke();
    const v = Gn(), m = f.reduce((S, w) => w.y < S.y || Math.abs(w.y - S.y) < 1e-6 && w.x > S.x ? w : S, f[0]);
    i.strokeStyle = "#0f172a", i.lineWidth = Math.max(2, o / 900) / a(X), i.beginPath(), i.moveTo(m.x, m.y), i.lineTo(v.x, v.y), i.stroke(), i.fillStyle = a(D) ? "#ffb000" : "#ffffff", i.strokeStyle = "#00ff66", i.lineWidth = Math.max(2, o / 900) / a(X), i.beginPath(), i.arc(v.x, v.y, Dn(), 0, Math.PI * 2), i.fill(), i.stroke(), i.restore();
  }
  function ja(i, o, f) {
    const v = H.get(o);
    if (!v) return;
    const m = fr(o, Ve(o)), S = Math.min(...m.map((W) => W.x)), w = Math.min(...m.map((W) => W.y)), I = String(v.view.label || "Label " + String(v.view.group_id)), N = 1 / Math.max(0.05, a(X));
    i.save(), i.font = "600 " + String(13 * N) + "px sans-serif";
    const L = i.measureText(I).width + 12 * N, R = Se(S, 2, Math.max(2, ve() - L - 2)), J = Se(w - 23 * N, 2, Math.max(2, Te() - 22 * N));
    i.fillStyle = f ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.62)", i.fillRect(R, J, L, 20 * N), i.fillStyle = "#ffffff", i.fillText(I, R + 6 * N, J + 14 * N), i.restore();
  }
  function vt() {
    var i, o, f;
    if (!h) return;
    const v = ve(), m = Te(), S = Ha(), w = Math.max(1, Math.round(v * S)), I = Math.max(1, Math.round(m * S));
    h.width !== w && (h.width = w), h.height !== I && (h.height = I);
    const N = h.getContext("2d");
    if (N && (N.setTransform(1, 0, 0, 1, 0, 0), N.clearRect(0, 0, h.width, h.height), Fr(N, S), Fa(N, v, m), a(g).enabled !== !1)) {
      if (Y()) {
        const L = Hn(), R = L.filter((W) => W !== a(M));
        L.includes(a(M)) && R.push(a(M));
        const J = zt();
        for (const W of R) {
          const q = H.get(W);
          if (q?.ready && q.tintCanvas) {
            const Oe = W === a(M) ? Se(Number((i = Ve(W).preview_alpha) !== null && i !== void 0 ? i : 0.22), 0, 1) : W === J ? 0.1 : 0.08;
            pr(N, S, q.tintCanvas, Ve(W), Oe);
          }
        }
        a(Me) === "overlap" && a(M) && J && L.includes(a(M)) && L.includes(J) && ka(N, S, a(M), J), Fr(N, S);
        for (const W of R)
          !((o = H.get(W)) === null || o === void 0) && o.ready && ja(N, W, W === a(M));
        a(M) && L.includes(a(M)) && (!((f = H.get(a(M))) === null || f === void 0) && f.ready) && kn(N, v);
        return;
      }
      B && a(A) && (pr(N, S, B, a(b)), Fr(N, S), kn(N, v));
    }
  }
  function me() {
    var i;
    return a(g).enabled ? Y() ? !!a(M) && !!(!((i = H.get(a(M))) === null || i === void 0) && i.ready) : a(A) && !!y : !1;
  }
  function Jt(i) {
    return !Y() || !H.has(i) || i === a(M) ? !1 : (a(M) && P.set(a(M), Ee(a(b))), _(M, i, !0), _(b, Ee(P.get(i)), !0), P.set(i, a(b)), It(), _(T, "当前 Label：" + pe()), be(), !0);
  }
  function jn(i) {
    _(Me, i, !0), _(T, i === "align" ? "对齐模式：默认只显示当前层" : "交叠预览：当前层突出、下一业务层淡显", !0), be();
  }
  function Va(i, o) {
    i.stopPropagation();
    const f = new Set(a(At)), v = new Set(a(Vt));
    lr(o) ? (f.add(o), v.delete(o)) : (f.delete(o), v.add(o)), _(At, f, !0), _(Vt, v, !0), _(T, "显示设置仅保存在浏览器，不改变保存数据"), be();
  }
  function Xa(i, o) {
    i.stopPropagation(), _(et, a(et) === o ? "" : o, !0), _(T, a(et) ? "已单独显示一层" : "已退出单层显示", !0), be();
  }
  function Wa() {
    _(X, 1), _(Fe, 0), _(ke, 0), _(T, "画布视图已复位（不影响保存数据）"), be();
  }
  function Vn(i) {
    const o = Se(a(X) * i, 0.2, 8), f = ve() / 2, v = Te() / 2, m = (f - a(Fe)) / a(X), S = (v - a(ke)) / a(X);
    _(Fe, f - m * o), _(ke, v - S * o), _(X, o, !0), _(T, "画布视图 " + Math.round(a(X) * 100) + "%（仅本地）"), be();
  }
  function za(i, o) {
    return i === "x" ? Object.assign(Object.assign({}, a(b)), { center_x: o }) : i === "y" ? Object.assign(Object.assign({}, a(b)), { center_y: o }) : i === "scale" ? Object.assign(Object.assign({}, a(b)), { scale: Se(o / 100, 0.01, 20) }) : i === "rotation" ? Object.assign(Object.assign({}, a(b)), { rotation_deg: ur(o) }) : Object.assign(Object.assign({}, a(b)), { preview_alpha: Se(o / 100, 0, 1) });
  }
  function jr(i) {
    const o = Number(a(rt)[i]);
    if (_(or, ""), !Number.isFinite(o)) {
      It();
      return;
    }
    const f = Ee(a(b)), v = Ee(za(i, o)), m = Math.abs(v.center_x - f.center_x) < 1e-9 && Math.abs(v.center_y - f.center_y) < 1e-9 && Math.abs(v.scale - f.scale) < 1e-12 && Math.abs(v.rotation_deg - f.rotation_deg) < 1e-9 && Math.abs(v.preview_alpha - f.preview_alpha) < 1e-12;
    if (Xe(v), m) {
      be();
      return;
    }
    dr("numeric", pe() + " 参数已更新");
  }
  function Ya(i, o) {
    _(rt, Object.assign(Object.assign({}, a(rt)), { [o]: i.currentTarget.value }), !0);
  }
  function qa(i, o) {
    const f = o.key === "ArrowUp" ? 1 : -1, v = i === "x" || i === "y" || i === "alpha" ? 1 : 0.1, m = i === "x" || i === "y" || i === "alpha" ? 0.1 : 0.01, S = i === "x" || i === "y" ? 10 : i === "alpha" ? 5 : 1;
    return f * (o.shiftKey ? m : o.altKey ? S : v);
  }
  function Za(i, o) {
    if (i.key === "Escape") {
      i.preventDefault(), _(or, ""), It(), i.currentTarget.blur();
      return;
    }
    if (i.key === "Enter") {
      i.preventDefault(), jr(o), i.currentTarget.blur();
      return;
    }
    if (i.key === "ArrowUp" || i.key === "ArrowDown") {
      i.preventDefault();
      const f = ae(a(rt)[o], 0);
      _(rt, Object.assign(Object.assign({}, a(rt)), { [o]: String(f + qa(o, i)) }), !0), jr(o);
    }
  }
  function Ja(i) {
    i.code === "Space" && !(i.target instanceof HTMLInputElement) && !(i.target instanceof HTMLTextAreaElement) && _(sr, !0);
  }
  function $a(i) {
    i.code === "Space" && _(sr, !1);
  }
  function Xn() {
    _(qe, ""), _(Be, ""), _(de, "before");
  }
  function Ka(i, o) {
    Ot() && (_(qe, o, !0), _(Be, ""), _(de, "before"), _(Ue, !1), i.dataTransfer && (i.dataTransfer.effectAllowed = "move", i.dataTransfer.setData("text/plain", o)));
  }
  function Qa(i, o) {
    if (!Ot() || !a(qe)) return;
    i.preventDefault(), i.dataTransfer && (i.dataTransfer.dropEffect = "move");
    const v = i.currentTarget.getBoundingClientRect();
    _(Be, o, !0), _(de, i.clientY < v.top + v.height / 2 ? "before" : "after", !0);
  }
  function es(i, o) {
    var f;
    if (!Ot()) return;
    i.preventDefault();
    const v = a(qe) || ((f = i.dataTransfer) === null || f === void 0 ? void 0 : f.getData("text/plain")) || "", m = a(Be) === o ? a(de) : "before", S = rs(v, o, m);
    Xn(), S && (_(Ue, !0), hr("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", !1, !0), setTimeout(
      () => {
        _(Ue, !1);
      },
      0
    ));
  }
  function ts() {
    Xn();
  }
  function rs(i, o, f) {
    if (!Ot() || i === o) return !1;
    const v = a(g).group_view, m = je(), S = m.findIndex((W) => F(W.group_id) === i), w = m.findIndex((W) => F(W.group_id) === o);
    if (!v || S < 0 || w < 0) return !1;
    const I = m.slice(), [N] = I.splice(S, 1);
    if (!N) return !1;
    const L = I.findIndex((W) => F(W.group_id) === o);
    if (L < 0) return !1;
    const R = L + (f === "after" ? 1 : 0);
    return I.splice(R, 0, N), I.some((W, q) => F(W.group_id) !== F(m[q].group_id)) ? (_(
      g,
      Object.assign(Object.assign({}, a(g)), {
        group_view: Object.assign(Object.assign({}, v), { groups: I })
      }),
      !0
    ), !0) : !1;
  }
  function Wn(i, o) {
    if (a(Ue)) {
      _(Ue, !1);
      return;
    }
    const f = i.target;
    f?.closest(".group-drag-handle") || Jt(o) && hr("select_group", "已选择 Label：" + pe(), !1, !0);
  }
  function ns(i) {
    const o = i.currentTarget.value;
    Jt(o) && hr("select_group", "已选择 Label：" + pe(), !1, !0);
  }
  function zn(i) {
    _(Mt, !0), _(re, "grabbing"), Xt = {
      client_x: i.clientX,
      client_y: i.clientY,
      pan_x: a(Fe),
      pan_y: a(ke)
    }, h.setPointerCapture(i.pointerId);
  }
  function is(i) {
    if (i.button !== 0) return;
    Ht();
    const o = Zt(i);
    if (a(sr)) {
      zn(i);
      return;
    }
    if (me() && Un(o.x, o.y)) {
      _(re, "grabbing"), _(D, !0), wt = {
        angle: Math.atan2(o.y - a(b).center_y, o.x - a(b).center_x) * 180 / Math.PI,
        rotation: Number(a(b).rotation_deg || 0)
      }, h.setPointerCapture(i.pointerId);
      return;
    }
    let f = !1;
    if (me() && Y()) {
      const v = kr(o.x, o.y);
      v && (Jt(v), f = !0);
    } else me() && (f = Ln(o.x, o.y));
    if (!f) {
      zn(i), _(T, "正在平移画布视图（仅本地）");
      return;
    }
    _(re, "grabbing"), _(C, !0), Re = {
      x: o.x,
      y: o.y,
      center_x: Number(a(b).center_x || 0),
      center_y: Number(a(b).center_y || 0)
    }, h.setPointerCapture(i.pointerId);
  }
  function Vr(i) {
    if (a(sr)) {
      _(re, "grab");
      return;
    }
    if (!me()) {
      _(re, "grab");
      return;
    }
    const o = Y() ? !!kr(i.x, i.y) : Ln(i.x, i.y);
    _(re, (Un(i.x, i.y) || o, "grab"), !0);
  }
  function as(i) {
    if (a(Mt)) {
      const f = h.getBoundingClientRect();
      _(Fe, Xt.pan_x + (i.clientX - Xt.client_x) * ve() / Math.max(1, f.width)), _(ke, Xt.pan_y + (i.clientY - Xt.client_y) * Te() / Math.max(1, f.height)), _(T, "正在平移画布视图（仅本地）"), be();
      return;
    }
    const o = Zt(i);
    if (!a(C) && !a(D)) {
      Vr(o);
      return;
    }
    if (_(re, "grabbing"), a(C))
      Xe(Object.assign(Object.assign({}, a(b)), {
        center_x: Re.center_x + o.x - Re.x,
        center_y: Re.center_y + o.y - Re.y
      })), _(T, "正在拖动 " + (Y() ? pe() : "版图") + "（本地）");
    else if (a(D)) {
      const f = Math.atan2(o.y - a(b).center_y, o.x - a(b).center_x) * 180 / Math.PI;
      Xe(Object.assign(Object.assign({}, a(b)), {
        rotation_deg: ur(wt.rotation + f - wt.angle)
      })), _(T, "正在旋转 " + (Y() ? pe() : "版图") + "（本地）");
    }
    be();
  }
  function ss() {
    !a(C) && !a(D) && !a(Mt) && _(re, "grab");
  }
  function Yn(i) {
    if (a(Mt)) {
      _(Mt, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      _(T, "画布视图已调整（仅本地，不影响保存数据）"), Vr(Zt(i));
      return;
    }
    if (a(C) || a(D)) {
      _(C, !1), _(D, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      Vr(Zt(i));
      const f = (Y() ? pe() : "Canvas") + " 已在本地更新，停止操作 250ms 后静默同步";
      Y() ? dr("canvas", f) : pt("canvas", f);
    }
  }
  function qn(i) {
    return i.deltaMode === WheelEvent.DOM_DELTA_LINE ? i.deltaY * 16 : i.deltaMode === WheelEvent.DOM_DELTA_PAGE ? i.deltaY * Math.max(1, h.getBoundingClientRect().height) : i.deltaY;
  }
  function os(i) {
    i.preventDefault();
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * ve(), v = (i.clientY - o.top) / Math.max(1, o.height) * Te();
    if (i.ctrlKey || i.metaKey) {
      const J = (f - a(Fe)) / a(X), W = (v - a(ke)) / a(X), q = Se(a(X) * Math.exp(-qn(i) * 15e-4), 0.2, 8);
      _(Fe, f - J * q), _(ke, v - W * q), _(X, q, !0), _(T, "画布视图 " + Math.round(a(X) * 100) + "%（仅本地）"), be();
      return;
    }
    if (!me()) return;
    const m = Zt(i);
    if (Y()) {
      const J = kr(m.x, m.y);
      J && Jt(J);
    }
    const S = Ur(m.x, m.y, a(b)), w = i.shiftKey ? 0.1 : i.altKey ? 5 : 1, I = Math.exp(-qn(i) * s * w), N = Se(Number(a(b).scale || 1) * I, 0.01, 20);
    let L = Object.assign(Object.assign({}, a(b)), { scale: N });
    const R = qt(S.x, S.y, L);
    L = Object.assign(Object.assign({}, L), {
      center_x: Number(L.center_x || 0) + m.x - R.x,
      center_y: Number(L.center_y || 0) + m.y - R.y
    }), Xe(L), dr("canvas", "本地缩放 " + (N * 100).toFixed(2) + "%；停止 250ms 后同步"), be();
  }
  function us() {
    me() && (Xe(Object.assign(Object.assign({}, a(b)), {
      center_x: ve() / 2,
      center_y: Te() / 2,
      scale: 1,
      rotation_deg: 0
    })), pt("reset", "居中归一 " + (Y() ? pe() : "版图") + "：scale=1，rotation=0"));
  }
  function ls() {
    me() && (Xe(Object.assign(Object.assign({}, a(b)), { center_x: ve() / 2, center_y: Te() / 2 })), pt("center", "居中 " + (Y() ? pe() : "版图") + "：保留缩放和旋转"));
  }
  function fs() {
    if (!me()) return;
    const [i, o, f, v] = In(), m = Math.max(1, f - i + 1), S = Math.max(1, v - o + 1), w = Se(Math.min(ve() / m, Te() / S) * 0.9, 0.01, 20);
    Xe(Object.assign(Object.assign({}, a(b)), {
      center_x: ve() / 2,
      center_y: Te() / 2,
      scale: w
    })), pt("fit", "适配 " + (Y() ? pe() : "版图") + "：scale=" + w.toFixed(3));
  }
  function cs() {
    if (!me()) return;
    const i = fr(), o = Math.min(...i.map((N) => N.x)), f = Math.max(...i.map((N) => N.x)), v = Math.min(...i.map((N) => N.y)), m = Math.max(...i.map((N) => N.y));
    let S = 0, w = 0;
    const I = Math.max(20, ve() * 0.03);
    if (f < I ? S = I - f : o > ve() - I && (S = ve() - I - o), m < I ? w = I - m : v > Te() - I && (w = Te() - I - v), S === 0 && w === 0) {
      _(T, (Y() ? pe() : "版图") + " 已经在视野内");
      return;
    }
    Xe(Object.assign(Object.assign({}, a(b)), {
      center_x: Number(a(b).center_x || 0) + S,
      center_y: Number(a(b).center_y || 0) + w
    })), pt("bring_into_view", "已找回 " + (Y() ? pe() : "版图") + " 到视野内");
  }
  j("keydown", fi, Ja), j("keyup", fi, $a);
  {
    let i = _t(() => a(C) || a(D) || a(Mt) ? "focus" : "base");
    Gl(e, {
      get visible() {
        return d.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return a(i);
      },
      padding: !1,
      get elem_id() {
        return d.shared.elem_id;
      },
      get elem_classes() {
        return d.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return d.shared.container;
      },
      get scale() {
        return d.shared.scale;
      },
      get min_width() {
        return d.shared.min_width;
      },
      children: (o, f) => {
        var v = Jl(), m = $(v);
        {
          var S = (oe) => {
            var _e = Vl(), ue = z($(_e), 2);
            _r(ue, 21, je, mr, (ot, ut) => {
              var Ce = Ul(), mt = $(Ce), se = {};
              Ke(
                ($t, Ie) => {
                  ye(mt, $t), se !== (se = Ie) && (Ce.value = (Ce.__value = Ie) ?? "");
                },
                [
                  () => a(ut).label || "Label " + String(a(ut).group_id),
                  () => F(a(ut).group_id)
                ]
              ), fe(ot, Ce);
            });
            var le;
            Ki(ue);
            var we = z(ue, 2), He = $(we);
            let at;
            var Ae = z(He, 2);
            let st;
            var gt = z(we, 2);
            {
              var $e = (ot) => {
                var ut = jl(), Ce = z($(ut), 2);
                _r(Ce, 21, je, mr, (mt, se, $t) => {
                  var Ie = kl();
                  let Qn;
                  var vr = $(Ie);
                  rr(vr, "draggable", !0);
                  var ei = z(vr, 2);
                  ei.textContent = $t + 1;
                  var ti = z(ei, 2), ri = $(ti), xs = z(ri);
                  {
                    var Es = (ne) => {
                      var Wr = Fl();
                      fe(ne, Wr);
                    }, Ss = _t(() => F(a(se).group_id) === a(M));
                    We(xs, (ne) => {
                      a(Ss) && ne(Es);
                    });
                  }
                  var gr = z(ti, 2), Ts = $(gr), ni = z(gr, 2);
                  let ii;
                  Ke(
                    (ne, Wr, ws, As, Ms) => {
                      Qn = yt(Ie, 1, "group-reorder-item svelte-r41nsf", null, Qn, ne), ye(ri, `${Wr ?? ""} `), rr(gr, "title", ws), ye(Ts, As), ii = yt(ni, 1, "solo-button svelte-r41nsf", null, ii, Ms);
                    },
                    [
                      () => ({
                        active: F(a(se).group_id) === a(M),
                        "drop-before": a(Be) === F(a(se).group_id) && a(de) === "before",
                        "drop-after": a(Be) === F(a(se).group_id) && a(de) === "after"
                      }),
                      () => a(se).label || "Label " + String(a(se).group_id),
                      () => lr(F(a(se).group_id)) ? "隐藏此层" : "显示此层",
                      () => lr(F(a(se).group_id)) ? "👁" : "—",
                      () => ({
                        active: a(et) === F(a(se).group_id)
                      })
                    ]
                  ), j("dragstart", vr, (ne) => Ka(ne, F(a(se).group_id))), j("dragend", vr, ts), j("click", gr, (ne) => Va(ne, F(a(se).group_id))), j("click", ni, (ne) => Xa(ne, F(a(se).group_id))), j("click", Ie, (ne) => Wn(ne, F(a(se).group_id))), j("keydown", Ie, (ne) => {
                    (ne.key === "Enter" || ne.key === " ") && (ne.preventDefault(), Wn(ne, F(a(se).group_id)));
                  }), j("dragover", Ie, (ne) => Qa(ne, F(a(se).group_id))), j("drop", Ie, (ne) => es(ne, F(a(se).group_id))), fe(mt, Ie);
                }), fe(ot, ut);
              }, bt = _t(() => Ot());
              We(gt, (ot) => {
                a(bt) && ot($e);
              });
            }
            Ke(() => {
              le !== (le = a(M)) && (ue.value = (ue.__value = a(M)) ?? "", Mr(ue, a(M))), at = yt(He, 1, "svelte-r41nsf", null, at, { active: a(Me) === "align" }), st = yt(Ae, 1, "svelte-r41nsf", null, st, { active: a(Me) === "overlap" });
            }), j("change", ue, ns), j("click", He, () => jn("align")), j("click", Ae, () => jn("overlap")), fe(oe, _e);
          }, w = _t(() => Y());
          We(m, (oe) => {
            a(w) && oe(S);
          });
        }
        var I = z(m, 2), N = $(I), L = z(N, 2), R = z(L, 2), J = z(R, 2), W = z(J, 2), q = z(W, 2), Oe = $(q), Ze = z(q, 2), Je = z(I, 2);
        {
          var hs = (oe) => {
            var _e = Wl();
            _r(
              _e,
              20,
              () => [
                ["x", "X", "px"],
                ["y", "Y", "px"],
                ["scale", "缩放", "%"],
                ["rotation", "旋转", "°"],
                ["alpha", "透明度", "%"]
              ],
              mr,
              (ue, le) => {
                var we = Xl(), He = $(we), at = $(He), Ae = z(He, 2), st = z(Ae, 2), gt = $(st);
                Ke(() => {
                  ye(at, le[1]), Yo(Ae, a(rt)[le[0]]), ye(gt, le[2]);
                }), j("focus", Ae, () => _(or, le[0], !0)), j("input", Ae, ($e) => Ya($e, le[0])), j("keydown", Ae, ($e) => Za($e, le[0])), j("blur", Ae, () => jr(le[0])), fe(ue, we);
              }
            ), fe(oe, _e);
          }, ds = _t(() => me());
          We(Je, (oe) => {
            a(ds) && oe(hs);
          });
        }
        var Zn = z(Je, 2);
        {
          var ps = (oe) => {
            var _e = ql();
            let ue;
            var le = $(_e), we = $(le), He = $(we), at = z(we, 2), Ae = $(at), st = z(le, 2);
            {
              var gt = (bt) => {
                var ot = Yl();
                _r(ot, 21, Pn, mr, (ut, Ce) => {
                  var mt = zl(), se = $(mt);
                  Ke(
                    ($t, Ie) => ye(se, `应用 ${a(Ce).candidate_id ?? ""}
								· 支持率 ${$t ?? ""}%
								· 分数 ${Ie ?? ""}`),
                    [
                      () => Math.round(ae(a(Ce).metrics?.central_support, 0) * 100),
                      () => ae(a(Ce).metrics?.score, 0).toFixed(3)
                    ]
                  ), j("click", mt, () => Bn(a(Ce))), fe(ut, mt);
                }), fe(bt, ot);
              }, $e = _t(() => Pn().length);
              We(st, (bt) => {
                a($e) && bt(gt);
              });
            }
            Ke(() => {
              ue = yt(_e, 1, "alignment-proposal svelte-r41nsf", null, ue, { ambiguous: a(ee).ambiguous === !0 }), ye(He, a(ee).ambiguous ? "存在周期歧义" : "粗对齐建议"), ye(Ae, a(ee).status);
            }), fe(oe, _e);
          };
          We(Zn, (oe) => {
            a(ee) && oe(ps);
          });
        }
        var Jn = z(Zn, 2), it = $(Jn);
        ra(it, (oe) => h = oe, () => h);
        var $n = z(Jn, 2);
        {
          var vs = (oe) => {
            var _e = Zl(), ue = $(_e), le = $(ue), we = z(ue, 2), He = $(we), at = z(we, 2), Ae = $(at);
            Ke(
              (st, gt, $e, bt) => {
                ye(le, `当前：${st ?? ""}`), ye(He, `下一层：${gt ?? ""}`), ye(Ae, `${$e ?? ""} × ${bt ?? ""}`);
              },
              [
                () => pe(),
                () => H.get(zt())?.view.label,
                () => pe(),
                () => H.get(zt())?.view.label
              ]
            ), fe(oe, _e);
          }, gs = _t(() => Y() && a(Me) === "overlap" && zt());
          We($n, (oe) => {
            a(gs) && oe(vs);
          });
        }
        var bs = z($n, 2), Xr = $(bs);
        let Kn;
        var ms = $(Xr), _s = z(Xr, 2), ys = $(_s);
        Ke(
          (oe, _e, ue, le, we, He) => {
            Ar(v, oe), N.disabled = _e, L.disabled = ue, R.disabled = le, J.disabled = we, ye(Oe, `${He ?? ""}%`), Ar(it, "cursor:" + a(re)), Kn = yt(Xr, 1, "svelte-r41nsf", null, Kn, { dirty: Pe > Ge }), ye(ms, Pe > Ge ? ce === null ? "本地修改" : "同步中" : "已同步"), ye(ys, a(T));
          },
          [
            () => "min-height:" + wa(d.props.height),
            () => !me(),
            () => !me(),
            () => !me(),
            () => !me(),
            () => Math.round(a(X) * 100)
          ]
        ), j("click", N, us), j("click", L, ls), j("click", R, fs), j("click", J, cs), j("click", W, () => Vn(0.8)), j("click", q, Wa), j("click", Ze, () => Vn(1.25)), j("pointerdown", it, is), j("pointermove", it, as), j("pointerup", it, Yn), j("pointercancel", it, Yn), j("pointerleave", it, ss), j("wheel", it, os), fe(o, v);
      },
      $$slots: { default: !0 }
    });
  }
  zi();
}
export {
  Kl as default
};
