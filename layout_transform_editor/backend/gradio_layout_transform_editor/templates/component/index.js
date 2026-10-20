import { i as yn, g as Li, o as Ns, n as Gt, u as ce, s as Ps, r as un, q as Ri, c as Gi, a as St, b as Pr, d as Ar, e as Bs, T as Cs, f as Ls, h as Mr, j as Tt, p as xn, k as wt, l as Di, m as Ui, t as Fi, v as Dt, w as ki, x as Ft, y as En, E as ji, z as a, A as ui, B as Rs, C as Vi, D as Xi, F as Sn, G as Gs, H as li, I as Ds, J as Us, K as Lt, L as Ze, M as Wi, N as Zr, O as Fs, P as ks, Q as js, R as Vs, S as zi, U as Xs, V as Yi, W as Tn, X as fi, Y as ci, Z as Ws, _ as zs, $ as Ys, a0 as qs, a1 as Zs, a2 as Br, a3 as Js, a4 as $s, a5 as Ks, a6 as Qs, a7 as eo, a8 as wn, a9 as to, aa as ro, ab as no, ac as io, ad as ao, ae as so, af as qi, ag as oo, ah as uo, ai as Be, aj as ln, ak as lo, al as fo, am as co, an as dt, ao as m, ap as ho, aq as po, ar as vo, as as go, at as bo, au as mo, av as Z, aw as _o, ax as hi, ay as yo, az as Jr, aA as Zi, aB as Ji, aC as Y, aD as V, aE as Ye, aF as xo, aG as di, aH as $, aI as xt, aJ as _e } from "./render-DlPxDIpa.js";
function $i(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const Eo = [];
function So(e, t = !1, r = !1) {
  return Sr(e, /* @__PURE__ */ new Map(), "", Eo, null, r);
}
function Sr(e, t, r, n, s = null, u = !1) {
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
    if (yn(e)) {
      var c = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, c), s !== null && t.set(s, c);
      for (var d = 0; d < e.length; d += 1) {
        var h = e[d];
        d in e && (c[d] = Sr(h, t, r, n, null, u));
      }
      return c;
    }
    if (Li(e) === Ns) {
      c = {}, t.set(e, c), s !== null && t.set(s, c);
      for (var p of Object.keys(e))
        c[p] = Sr(
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
      return Sr(
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
function To(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), Gt;
  const n = ce(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Ct = [];
function wo(e, t) {
  return {
    subscribe: Cr(e, t).subscribe
  };
}
function Cr(e, t = Gt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function s(c) {
    if (Ps(e, c) && (e = c, r)) {
      const d = !Ct.length;
      for (const h of n)
        h[1](), Ct.push(h, e);
      if (d) {
        for (let h = 0; h < Ct.length; h += 2)
          Ct[h][0](Ct[h + 1]);
        Ct.length = 0;
      }
    }
  }
  function u(c) {
    s(c(
      /** @type {T} */
      e
    ));
  }
  function l(c, d = Gt) {
    const h = [c, d];
    return n.add(h), n.size === 1 && (r = t(s, u) || Gt), c(
      /** @type {T} */
      e
    ), () => {
      n.delete(h), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: s, update: u, subscribe: l };
}
function Xt(e, t, r) {
  const n = !Array.isArray(e), s = n ? [e] : e;
  if (!s.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const u = t.length < 2;
  return wo(r, (l, c) => {
    let d = !1;
    const h = [];
    let p = 0, E = Gt;
    const y = () => {
      if (p)
        return;
      E();
      const I = t(n ? h[0] : h, l, c);
      u ? l(I) : E = typeof I == "function" ? I : Gt;
    }, B = s.map(
      (I, P) => To(
        I,
        (O) => {
          h[P] = O, p &= ~(1 << P), d && y();
        },
        () => {
          p |= 1 << P;
        }
      )
    );
    return d = !0, y(), function() {
      un(B), E(), d = !1;
    };
  });
}
let _r = !1;
function Ao(e) {
  var t = _r;
  try {
    return _r = !1, [e(), _r];
  } finally {
    _r = t;
  }
}
function Mo(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Ri(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Oo = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Ho(e) {
  return (
    /** @type {string} */
    Oo?.createHTML(e) ?? e
  );
}
function Ki(e) {
  var t = Gi("template");
  return t.innerHTML = Ho(e.replaceAll("<!>", "<!---->")), t.content;
}
function ar(e, t) {
  var r = (
    /** @type {Effect} */
    Pr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function ye(e, t) {
  var r = (t & Cs) !== 0, n = (t & Ls) !== 0, s, u = !e.startsWith("<!>");
  return () => {
    s === void 0 && (s = Ki(u ? e : "<!>" + e), r || (s = /** @type {TemplateNode} */
    Ar(s)));
    var l = (
      /** @type {TemplateNode} */
      n || Bs ? document.importNode(s, !0) : s.cloneNode(!0)
    );
    if (r) {
      var c = (
        /** @type {TemplateNode} */
        Ar(l)
      ), d = (
        /** @type {TemplateNode} */
        l.lastChild
      );
      ar(c, d);
    } else
      ar(l, l);
    return l;
  };
}
// @__NO_SIDE_EFFECTS__
function Io(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), s = `<${r}>${n ? e : "<!>" + e}</${r}>`, u;
  return () => {
    if (!u) {
      var l = (
        /** @type {DocumentFragment} */
        Ki(s)
      ), c = (
        /** @type {Element} */
        Ar(l)
      );
      u = /** @type {Element} */
      Ar(c);
    }
    var d = (
      /** @type {TemplateNode} */
      u.cloneNode(!0)
    );
    return ar(d, d), d;
  };
}
// @__NO_SIDE_EFFECTS__
function No(e, t) {
  return /* @__PURE__ */ Io(e, t, "svg");
}
function Po() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = St();
  return e.append(t, r), ar(t, r), e;
}
function le(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Qi {
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
        Mr(n), this.#n.delete(r);
      else {
        var s = this.#e.get(r);
        s && (Mr(s.effect), this.#r.set(r, s.effect), this.#e.delete(r), s.fragment.lastChild.remove(), this.anchor.before(s.fragment), n = s.effect);
      }
      for (const [u, l] of this.#t) {
        if (this.#t.delete(u), u === t)
          break;
        const c = this.#e.get(l);
        c && (Tt(c.effect), this.#e.delete(l));
      }
      for (const [u, l] of this.#r) {
        if (u === r || this.#n.has(u)) continue;
        const c = () => {
          if (Array.from(this.#t.values()).includes(u)) {
            var h = document.createDocumentFragment();
            Ui(l, h), h.append(St()), this.#e.set(u, { effect: l, fragment: h });
          } else
            Tt(l);
          this.#n.delete(u), this.#r.delete(u);
        };
        this.#i || !n ? (this.#n.add(u), xn(l, c, !1)) : c();
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
      r.includes(n) || (Tt(s.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Di
    ), s = Fi();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (s) {
        var u = document.createDocumentFragment(), l = St();
        u.append(l), this.#e.set(t, {
          effect: wt(() => r(l)),
          fragment: u
        });
      } else
        this.#r.set(
          t,
          wt(() => r(this.anchor))
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
function Bo(e) {
  Dt === null && $i(), ki && Dt.l !== null ? Lo(Dt).m.push(e) : Ft(() => {
    const t = ce(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Co(e) {
  Dt === null && $i(), Bo(() => () => ce(e));
}
function Lo(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function qe(e, t, r = !1) {
  var n = new Qi(e), s = r ? ji : 0;
  function u(l, c) {
    n.ensure(l, c);
  }
  En(() => {
    var l = !1;
    t((c, d = 0) => {
      l = !0, u(d, c);
    }), l || u(-1, null);
  }, s);
}
function yr(e, t) {
  return t;
}
function Ro(e, t, r) {
  for (var n = [], s = t.length, u, l = t.length, c = 0; c < s; c++) {
    let E = t[c];
    xn(
      E,
      () => {
        if (u) {
          if (u.pending.delete(E), u.done.add(E), u.pending.size === 0) {
            var y = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            fn(e, Sn(u.done)), y.delete(u), y.size === 0 && (e.outrogroups = null);
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
      js(p), p.append(h), e.items.clear();
    }
    fn(e, t, !d);
  } else
    u = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(u);
}
function fn(e, t, r = !0) {
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
      u.f |= Ze;
      const l = document.createDocumentFragment();
      Ui(u, l);
    } else
      Tt(t[s], r);
  }
}
var pi;
function xr(e, t, r, n, s, u = null) {
  var l = e, c = /* @__PURE__ */ new Map(), d = (t & Vi) !== 0;
  if (d) {
    var h = (
      /** @type {Element} */
      e
    );
    l = h.appendChild(St());
  }
  var p = null, E = Xi(() => {
    var H = r();
    return (
      /** @type {V[]} */
      yn(H) ? H : H == null ? [] : Sn(H)
    );
  }), y, B = /* @__PURE__ */ new Map(), I = !0;
  function P(H) {
    (M.effect.f & Wi) === 0 && (M.pending.delete(H), M.fallback = p, Go(M, y, l, t, n), p !== null && (y.length === 0 ? (p.f & Ze) === 0 ? Mr(p) : (p.f ^= Ze, nr(p, null, l)) : xn(p, () => {
      p = null;
    })));
  }
  function O(H) {
    M.pending.delete(H);
  }
  var S = En(() => {
    y = /** @type {V[]} */
    a(E);
    for (var H = y.length, _ = /* @__PURE__ */ new Set(), b = (
      /** @type {Batch} */
      Di
    ), A = Fi(), T = 0; T < H; T += 1) {
      var F = y[T], ee = n(F, T), G = I ? null : c.get(ee);
      G ? (G.v && ui(G.v, F), G.i && ui(G.i, T), A && b.unskip_effect(G.e)) : (G = Do(
        c,
        I ? l : pi ??= St(),
        F,
        ee,
        T,
        s,
        t,
        r
      ), I || (G.e.f |= Ze), c.set(ee, G)), _.add(ee);
    }
    if (H === 0 && u && !p && (I ? p = wt(() => u(l)) : (p = wt(() => u(pi ??= St())), p.f |= Ze)), H > _.size && Rs(), !I)
      if (B.set(b, _), A) {
        for (const [vt, Ce] of c)
          _.has(vt) || b.skip_effect(Ce.e);
        b.oncommit(P), b.ondiscard(O);
      } else
        P(b);
    a(E);
  }), M = { effect: S, items: c, pending: B, outrogroups: null, fallback: p };
  I = !1;
}
function tr(e) {
  for (; e !== null && (e.f & Fs) === 0; )
    e = e.next;
  return e;
}
function Go(e, t, r, n, s) {
  var u = (n & ks) !== 0, l = t.length, c = e.items, d = tr(e.effect.first), h, p = null, E, y = [], B = [], I, P, O, S;
  if (u)
    for (S = 0; S < l; S += 1)
      I = t[S], P = s(I, S), O = /** @type {EachItem} */
      c.get(P).e, (O.f & Ze) === 0 && (O.nodes?.a?.measure(), (E ??= /* @__PURE__ */ new Set()).add(O));
  for (S = 0; S < l; S += 1) {
    if (I = t[S], P = s(I, S), O = /** @type {EachItem} */
    c.get(P).e, e.outrogroups !== null)
      for (const G of e.outrogroups)
        G.pending.delete(O), G.done.delete(O);
    if ((O.f & Zr) !== 0 && (Mr(O), u && (O.nodes?.a?.unfix(), (E ??= /* @__PURE__ */ new Set()).delete(O))), (O.f & Ze) !== 0)
      if (O.f ^= Ze, O === d)
        nr(O, null, r);
      else {
        var M = p ? p.next : d;
        O === e.effect.last && (e.effect.last = O.prev), O.prev && (O.prev.next = O.next), O.next && (O.next.prev = O.prev), ht(e, p, O), ht(e, O, M), nr(O, M, r), p = O, y = [], B = [], d = tr(p.next);
        continue;
      }
    if (O !== d) {
      if (h !== void 0 && h.has(O)) {
        if (y.length < B.length) {
          var H = B[0], _;
          p = H.prev;
          var b = y[0], A = y[y.length - 1];
          for (_ = 0; _ < y.length; _ += 1)
            nr(y[_], H, r);
          for (_ = 0; _ < B.length; _ += 1)
            h.delete(B[_]);
          ht(e, b.prev, A.next), ht(e, p, b), ht(e, A, H), d = H, p = A, S -= 1, y = [], B = [];
        } else
          h.delete(O), nr(O, d, r), ht(e, O.prev, O.next), ht(e, O, p === null ? e.effect.first : p.next), ht(e, p, O), p = O;
        continue;
      }
      for (y = [], B = []; d !== null && d !== O; )
        (h ??= /* @__PURE__ */ new Set()).add(d), B.push(d), d = tr(d.next);
      if (d === null)
        continue;
    }
    (O.f & Ze) === 0 && y.push(O), p = O, d = tr(O.next);
  }
  if (e.outrogroups !== null) {
    for (const G of e.outrogroups)
      G.pending.size === 0 && (fn(e, Sn(G.done)), e.outrogroups?.delete(G));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (d !== null || h !== void 0) {
    var T = [];
    if (h !== void 0)
      for (O of h)
        (O.f & Zr) === 0 && T.push(O);
    for (; d !== null; )
      (d.f & Zr) === 0 && d !== e.fallback && T.push(d), d = tr(d.next);
    var F = T.length;
    if (F > 0) {
      var ee = (n & Vi) !== 0 && l === 0 ? r : null;
      if (u) {
        for (S = 0; S < F; S += 1)
          T[S].nodes?.a?.measure();
        for (S = 0; S < F; S += 1)
          T[S].nodes?.a?.fix();
      }
      Ro(e, T, ee);
    }
  }
  u && Ri(() => {
    if (E !== void 0)
      for (O of E)
        O.nodes?.a?.apply();
  });
}
function Do(e, t, r, n, s, u, l, c) {
  var d = (l & Ds) !== 0 ? (l & Us) === 0 ? Lt(r, !1, !1) : li(r) : null, h = (l & Gs) !== 0 ? li(s) : null;
  return {
    v: d,
    i: h,
    e: wt(() => (u(t, d ?? r, h ?? s, c), () => {
      e.delete(n);
    }))
  };
}
function nr(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, s = e.nodes.end, u = t && (t.f & Ze) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var l = (
        /** @type {TemplateNode} */
        Vs(n)
      );
      if (u.before(n), n === s)
        return;
      n = l;
    }
}
function ht(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Uo(e, t, r, n, s) {
  var u = t.$$slots?.[r], l = !1;
  u === !0 && (u = t.children, l = !0), u === void 0 || u(e, l ? () => n : n);
}
function Fo(e, t, r, n, s, u) {
  var l = null, c = (
    /** @type {TemplateNode} */
    e
  ), d = new Qi(c, !1);
  En(() => {
    const h = t() || null;
    var p = h === "svg" ? Xs : void 0;
    if (h === null) {
      d.ensure(null, null);
      return;
    }
    return d.ensure(h, (E) => {
      if (h) {
        if (l = Gi(h, p), ar(l, l), n) {
          var y = null, B = l.appendChild(St());
          n(l, B), y?.remove();
        }
        Pr.nodes.end = l, E.before(l);
      }
    }), () => {
    };
  }, ji), zi(() => {
  });
}
function ko(e, t) {
  var r = void 0, n;
  Yi(() => {
    r !== (r = t()) && (n && (Tt(n), n = null), r && (n = wt(() => {
      Tn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function ea(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var s = e.length;
    for (t = 0; t < s; t++) e[t] && (r = ea(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function jo() {
  for (var e, t, r = 0, n = "", s = arguments.length; r < s; r++) (e = arguments[r]) && (t = ea(e)) && (n && (n += " "), n += t);
  return n;
}
function Vo(e) {
  return typeof e == "object" ? jo(e) : e ?? "";
}
const vi = [...` 	
\r\f \v\uFEFF`];
function Xo(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var s of Object.keys(r))
      if (r[s])
        n = n ? n + " " + s : s;
      else if (n.length)
        for (var u = s.length, l = 0; (l = n.indexOf(s, l)) >= 0; ) {
          var c = l + u;
          (l === 0 || vi.includes(n[l - 1])) && (c === n.length || vi.includes(n[c])) ? n = (l === 0 ? "" : n.substring(0, l)) + n.substring(c + 1) : l = c;
        }
  }
  return n === "" ? null : n;
}
function gi(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var s of Object.keys(e)) {
    var u = e[s];
    u != null && u !== "" && (n += " " + s + ": " + u + r);
  }
  return n;
}
function $r(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Wo(e, t) {
  if (t) {
    var r = "", n, s;
    if (Array.isArray(t) ? (n = t[0], s = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var u = !1, l = 0, c = !1, d = [];
      n && d.push(...Object.keys(n).map($r)), s && d.push(...Object.keys(s).map($r));
      var h = 0, p = -1;
      const P = e.length;
      for (var E = 0; E < P; E++) {
        var y = e[E];
        if (c ? y === "/" && e[E - 1] === "*" && (c = !1) : u ? u === y && (u = !1) : y === "/" && e[E + 1] === "*" ? c = !0 : y === '"' || y === "'" ? u = y : y === "(" ? l++ : y === ")" && l--, !c && u === !1 && l === 0) {
          if (y === ":" && p === -1)
            p = E;
          else if (y === ";" || E === P - 1) {
            if (p !== -1) {
              var B = $r(e.substring(h, p).trim());
              if (!d.includes(B)) {
                y !== ";" && E++;
                var I = e.substring(h, E).trim();
                r += " " + I + ";";
              }
            }
            h = E + 1, p = -1;
          }
        }
      }
    }
    return n && (r += gi(n)), s && (r += gi(s, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Et(e, t, r, n, s, u) {
  var l = (
    /** @type {any} */
    e[fi]
  );
  if (l !== r || l === void 0) {
    var c = Xo(r, n, u);
    c == null ? e.removeAttribute("class") : t ? e.className = c : e.setAttribute("class", c), e[fi] = r;
  } else if (u && s !== u)
    for (var d in u) {
      var h = !!u[d];
      (s == null || h !== !!s[d]) && e.classList.toggle(d, h);
    }
  return u;
}
function Kr(e, t = {}, r, n) {
  for (var s in r) {
    var u = r[s];
    t[s] !== u && (r[s] == null ? e.style.removeProperty(s) : e.style.setProperty(s, u, n));
  }
}
function Or(e, t, r, n) {
  var s = (
    /** @type {any} */
    e[ci]
  );
  if (s !== t) {
    var u = Wo(t, n);
    u == null ? e.removeAttribute("style") : e.style.cssText = u, e[ci] = t;
  } else n && (Array.isArray(n) ? (Kr(e, r?.[0], n[0]), Kr(e, r?.[1], n[1], "important")) : Kr(e, r, n));
  return n;
}
function Hr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!yn(t))
      return Ws();
    for (var n of e.options)
      n.selected = t.includes(bi(n));
    return;
  }
  for (n of e.options) {
    var s = bi(n);
    if (zs(s, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function ta(e) {
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
  }), zi(() => {
    t.disconnect();
  });
}
function bi(e) {
  return "__value" in e ? e.__value : e.value;
}
const ir = /* @__PURE__ */ Symbol("class"), Rt = /* @__PURE__ */ Symbol("style"), ra = /* @__PURE__ */ Symbol("is custom element"), na = /* @__PURE__ */ Symbol("is html"), zo = Br ? "input" : "INPUT", Yo = Br ? "option" : "OPTION", qo = Br ? "select" : "SELECT", Zo = Br ? "progress" : "PROGRESS";
function Jo(e, t) {
  var r = An(e);
  r.value === (r.value = // treat null and undefined the same for the initial value
  t ?? void 0) || // @ts-expect-error
  // `progress` elements always need their value set when it's `0`
  e.value === t && (t !== 0 || e.nodeName !== Zo) || (e.value = t ?? "");
}
function $o(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function Ut(e, t, r, n) {
  var s = An(e);
  s[t] !== (s[t] = r) && (t === "loading" && (e[Ys] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ia(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Ko(e, t, r, n, s = !1, u = !1) {
  var l = An(e), c = l[ra], d = !l[na], h = t || {}, p = e.nodeName === Yo;
  for (var E in t)
    E in r || (r[E] = null);
  r.class ? r.class = Vo(r.class) : r.class = null, r[Rt] && (r.style ??= null);
  var y = ia(e);
  if (e.nodeName === zo && "type" in r && ("value" in r || "__value" in r)) {
    var B = r.type;
    (B !== h.type || B === void 0 && e.hasAttribute("type")) && (h.type = B, Ut(e, "type", B));
  }
  for (const _ in r) {
    let b = r[_];
    if (p && _ === "value" && b == null) {
      e.value = e.__value = "", h[_] = b;
      continue;
    }
    if (_ === "class") {
      var I = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Et(e, I, b, n, t?.[ir], r[ir]), h[_] = b, h[ir] = r[ir];
      continue;
    }
    if (_ === "style") {
      Or(e, b, t?.[Rt], r[Rt]), h[_] = b, h[Rt] = r[Rt];
      continue;
    }
    var P = h[_];
    if (!(b === P && !(b === void 0 && e.hasAttribute(_)))) {
      h[_] = b;
      var O = _[0] + _[1];
      if (O !== "$$")
        if (O === "on") {
          const A = {}, T = "$$" + _;
          let F = _.slice(2);
          var S = io(F);
          if (Qs(F) && (F = F.slice(0, -7), A.capture = !0), !S && P) {
            if (b != null) continue;
            e.removeEventListener(F, h[T], A), h[T] = null;
          }
          if (S)
            eo(F, e, b), wn([F]);
          else if (b != null) {
            let ee = function(G) {
              h[_].call(this, G);
            };
            h[T] = to(F, e, ee, A);
          }
        } else if (_ === "style")
          Ut(e, _, b);
        else if (_ === "autofocus")
          Mo(
            /** @type {HTMLElement} */
            e,
            !!b
          );
        else if (!c && (_ === "__value" || _ === "value" && b != null))
          e.value = e.__value = b;
        else if (_ === "selected" && p)
          $o(
            /** @type {HTMLOptionElement} */
            e,
            b
          );
        else {
          var M = _;
          d || (M = ro(M));
          var H = M === "defaultValue" || M === "defaultChecked";
          if (b == null && !c && !H)
            if (l[_] = null, M === "value" || M === "checked") {
              let A = (
                /** @type {HTMLInputElement} */
                e
              );
              const T = t === void 0;
              if (M === "value") {
                let F = A.defaultValue;
                A.removeAttribute(M), A.defaultValue = F, A.value = A.__value = T ? F : null;
              } else {
                let F = A.defaultChecked;
                A.removeAttribute(M), A.defaultChecked = F, A.checked = T ? F : !1;
              }
            } else
              e.removeAttribute(_);
          else H || y.includes(M) && (c || typeof b != "string") ? (e[M] = b, M in l && (l[M] = no)) : typeof b != "function" && Ut(e, M, b);
        }
    }
  }
  return h;
}
function Qo(e, t, r = [], n = [], s = [], u, l = !1, c = !1) {
  $s(s, r, n, (d) => {
    var h = void 0, p = {}, E = e.nodeName === qo, y = !1;
    if (Yi(() => {
      var I = t(...d.map(a)), P = Ko(
        e,
        h,
        I,
        u,
        l,
        c
      );
      y && E && "value" in I && Hr(
        /** @type {HTMLSelectElement} */
        e,
        I.value
      );
      for (let S of Object.getOwnPropertySymbols(p))
        I[S] || Tt(p[S]);
      for (let S of Object.getOwnPropertySymbols(I)) {
        var O = I[S];
        S.description === Ks && (!h || O !== h[S]) && (p[S] && Tt(p[S]), p[S] = wt(() => ko(e, () => O))), P[S] = O;
      }
      h = P;
    }), E) {
      var B = (
        /** @type {HTMLSelectElement} */
        e
      );
      Tn(() => {
        Hr(
          B,
          /** @type {Record<string | symbol, any>} */
          h.value,
          !0
        ), ta(B);
      });
    }
    y = !0;
  });
}
function An(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[qs] ??= {
      [ra]: e.nodeName.includes("-"),
      [na]: e.namespaceURI === Zs
    }
  );
}
var mi = /* @__PURE__ */ new Map();
function ia(e) {
  var t = e.getAttribute("is") || e.nodeName, r = mi.get(t);
  if (r) return r;
  mi.set(t, r = []);
  for (var n, s = e, u = Element.prototype; u !== s; ) {
    n = Js(s);
    for (var l in n)
      n[l].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      l !== "innerHTML" && l !== "textContent" && l !== "innerText" && r.push(l);
    s = Li(s);
  }
  return r;
}
function Qr(e, t) {
  return e === t || e?.[qi] === t;
}
function aa(e = {}, t, r, n) {
  var s = (
    /** @type {ComponentContext} */
    Dt.r
  ), u = (
    /** @type {Effect} */
    Pr
  );
  return Tn(() => {
    var l, c;
    return ao(() => {
      l = c, c = [], ce(() => {
        Qr(r(...c), e) || (t(e, ...c), l && Qr(r(...l), e) && t(null, ...l));
      });
    }), () => {
      let d = u;
      for (; d !== s && d.parent !== null && d.parent.f & so; )
        d = d.parent;
      const h = () => {
        c && Qr(r(...c), e) && t(null, ...c);
      }, p = d.teardown;
      d.teardown = () => {
        h(), p?.();
      };
    };
  }), e;
}
function eu(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Dt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Be(t.s);
  if (e) {
    let s = 0, u = (
      /** @type {Record<string, any>} */
      {}
    );
    const l = ln(() => {
      let c = !1;
      const d = t.s;
      for (const h in d)
        d[h] !== u[h] && (u[h] = d[h], c = !0);
      return c && s++, s;
    });
    n = () => a(l);
  }
  r.b.length && oo(() => {
    _i(t, n), un(r.b);
  }), Ft(() => {
    const s = ce(() => r.m.map(uo));
    return () => {
      for (const u of s)
        typeof u == "function" && u();
    };
  }), r.a.length && Ft(() => {
    _i(t, n), un(r.a);
  });
}
function _i(e, t) {
  if (e.l.s)
    for (const r of e.l.s) a(r);
  t();
}
const tu = {
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
function ru(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    tu
  );
}
function Q(e, t, r, n) {
  var s = !ki || (r & po) !== 0, u = (r & ho) !== 0, l = (r & go) !== 0, c = (
    /** @type {V} */
    n
  ), d = !0, h = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), p = () => l && s ? (h ??= ln(
    /** @type {() => V} */
    n
  ), a(h)) : (d && (d = !1, c = l ? ce(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), c);
  let E;
  if (u) {
    var y = qi in e || mo in e;
    E = lo(e, t)?.set ?? (y && t in e ? (_) => e[t] = _ : void 0);
  }
  var B, I = !1;
  u ? [B, I] = Ao(() => (
    /** @type {V} */
    e[t]
  )) : B = /** @type {V} */
  e[t], B === void 0 && n !== void 0 && (B = p(), E && (s && fo(), E(B)));
  var P;
  if (s ? P = () => {
    var _ = (
      /** @type {V} */
      e[t]
    );
    return _ === void 0 ? p() : (d = !0, _);
  } : P = () => {
    var _ = (
      /** @type {V} */
      e[t]
    );
    return _ !== void 0 && (c = /** @type {V} */
    void 0), _ === void 0 ? c : _;
  }, s && (r & co) === 0)
    return P;
  if (E) {
    var O = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(_, b) {
        return arguments.length > 0 ? ((!s || !b || O || I) && E(b ? P() : _), _) : P();
      })
    );
  }
  var S = !1, M = ((r & vo) !== 0 ? ln : Xi)(() => (S = !1, P()));
  u && a(M);
  var H = (
    /** @type {Effect} */
    Pr
  );
  return (
    /** @type {() => V} */
    (function(_, b) {
      if (arguments.length > 0) {
        const A = b ? a(M) : s && u ? dt(_) : _;
        return m(M, A), S = !0, c !== void 0 && (c = A), _;
      }
      return bo && S || (H.f & Wi) !== 0 ? M.v : a(M);
    })
  );
}
const nu = [
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
], yi = {
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
nu.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: yi[t][r],
    secondary: yi[t][n]
  }
}), {});
function iu(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var en, xi;
function au() {
  if (xi) return en;
  xi = 1;
  var e = function(M) {
    return t(M) && !r(M);
  };
  function t(S) {
    return !!S && typeof S == "object";
  }
  function r(S) {
    var M = Object.prototype.toString.call(S);
    return M === "[object RegExp]" || M === "[object Date]" || u(S);
  }
  var n = typeof Symbol == "function" && Symbol.for, s = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function u(S) {
    return S.$$typeof === s;
  }
  function l(S) {
    return Array.isArray(S) ? [] : {};
  }
  function c(S, M) {
    return M.clone !== !1 && M.isMergeableObject(S) ? P(l(S), S, M) : S;
  }
  function d(S, M, H) {
    return S.concat(M).map(function(_) {
      return c(_, H);
    });
  }
  function h(S, M) {
    if (!M.customMerge)
      return P;
    var H = M.customMerge(S);
    return typeof H == "function" ? H : P;
  }
  function p(S) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(S).filter(function(M) {
      return Object.propertyIsEnumerable.call(S, M);
    }) : [];
  }
  function E(S) {
    return Object.keys(S).concat(p(S));
  }
  function y(S, M) {
    try {
      return M in S;
    } catch {
      return !1;
    }
  }
  function B(S, M) {
    return y(S, M) && !(Object.hasOwnProperty.call(S, M) && Object.propertyIsEnumerable.call(S, M));
  }
  function I(S, M, H) {
    var _ = {};
    return H.isMergeableObject(S) && E(S).forEach(function(b) {
      _[b] = c(S[b], H);
    }), E(M).forEach(function(b) {
      B(S, b) || (y(S, b) && H.isMergeableObject(M[b]) ? _[b] = h(b, H)(S[b], M[b], H) : _[b] = c(M[b], H));
    }), _;
  }
  function P(S, M, H) {
    H = H || {}, H.arrayMerge = H.arrayMerge || d, H.isMergeableObject = H.isMergeableObject || e, H.cloneUnlessOtherwiseSpecified = c;
    var _ = Array.isArray(M), b = Array.isArray(S), A = _ === b;
    return A ? _ ? H.arrayMerge(S, M, H) : I(S, M, H) : c(M, H);
  }
  P.all = function(M, H) {
    if (!Array.isArray(M))
      throw new Error("first argument should be an array");
    return M.reduce(function(_, b) {
      return P(_, b, H);
    }, {});
  };
  var O = P;
  return en = O, en;
}
var su = au();
const ou = /* @__PURE__ */ iu(su);
var cn = function(e, t) {
  return cn = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var s in n) Object.prototype.hasOwnProperty.call(n, s) && (r[s] = n[s]);
  }, cn(e, t);
};
function Lr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  cn(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var X = function() {
  return X = Object.assign || function(t) {
    for (var r, n = 1, s = arguments.length; n < s; n++) {
      r = arguments[n];
      for (var u in r) Object.prototype.hasOwnProperty.call(r, u) && (t[u] = r[u]);
    }
    return t;
  }, X.apply(this, arguments);
};
function uu(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var s = 0, n = Object.getOwnPropertySymbols(e); s < n.length; s++)
      t.indexOf(n[s]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[s]) && (r[n[s]] = e[n[s]]);
  return r;
}
function tn(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, s = t.length, u; n < s; n++)
    (u || !(n in t)) && (u || (u = Array.prototype.slice.call(t, 0, n)), u[n] = t[n]);
  return e.concat(u || Array.prototype.slice.call(t));
}
function rn(e, t) {
  var r = t && t.cache ? t.cache : vu, n = t && t.serializer ? t.serializer : du, s = t && t.strategy ? t.strategy : cu;
  return s(e, {
    cache: r,
    serializer: n
  });
}
function lu(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function fu(e, t, r, n) {
  var s = lu(n) ? n : r(n), u = t.get(s);
  return typeof u > "u" && (u = e.call(this, n), t.set(s, u)), u;
}
function sa(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), s = r(n), u = t.get(s);
  return typeof u > "u" && (u = e.apply(this, n), t.set(s, u)), u;
}
function oa(e, t, r, n, s) {
  return r.bind(t, e, n, s);
}
function cu(e, t) {
  var r = e.length === 1 ? fu : sa;
  return oa(e, this, r, t.cache.create(), t.serializer);
}
function hu(e, t) {
  return oa(e, this, sa, t.cache.create(), t.serializer);
}
var du = function() {
  return JSON.stringify(arguments);
}, pu = (
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
), vu = {
  create: function() {
    return new pu();
  }
}, nn = {
  variadic: hu
}, D;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(D || (D = {}));
var K;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(K || (K = {}));
var kt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(kt || (kt = {}));
function Ei(e) {
  return e.type === K.literal;
}
function gu(e) {
  return e.type === K.argument;
}
function ua(e) {
  return e.type === K.number;
}
function la(e) {
  return e.type === K.date;
}
function fa(e) {
  return e.type === K.time;
}
function ca(e) {
  return e.type === K.select;
}
function ha(e) {
  return e.type === K.plural;
}
function bu(e) {
  return e.type === K.pound;
}
function da(e) {
  return e.type === K.tag;
}
function pa(e) {
  return !!(e && typeof e == "object" && e.type === kt.number);
}
function hn(e) {
  return !!(e && typeof e == "object" && e.type === kt.dateTime);
}
var va = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, mu = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function _u(e) {
  var t = {};
  return e.replace(mu, function(r) {
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
var yu = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function xu(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(yu).filter(function(y) {
    return y.length > 0;
  }), r = [], n = 0, s = t; n < s.length; n++) {
    var u = s[n], l = u.split("/");
    if (l.length === 0)
      throw new Error("Invalid number skeleton");
    for (var c = l[0], d = l.slice(1), h = 0, p = d; h < p.length; h++) {
      var E = p[h];
      if (E.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: c, options: d });
  }
  return r;
}
function Eu(e) {
  return e.replace(/^(.*?)-/, "");
}
var Si = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, ga = /^(@+)?(\+|#+)?[rs]?$/g, Su = /(\*)(0+)|(#+)(0+)|(0+)/g, ba = /^(0+)$/;
function Ti(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(ga, function(r, n, s) {
    return typeof s != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : s === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof s == "string" ? s.length : 0)), "";
  }), t;
}
function ma(e) {
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
function Tu(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !ba.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function wi(e) {
  var t = {}, r = ma(e);
  return r || t;
}
function wu(e) {
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
        t.style = "unit", t.unit = Eu(s.options[0]);
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
        t = X(X(X({}, t), { notation: "scientific" }), s.options.reduce(function(d, h) {
          return X(X({}, d), wi(h));
        }, {}));
        continue;
      case "engineering":
        t = X(X(X({}, t), { notation: "engineering" }), s.options.reduce(function(d, h) {
          return X(X({}, d), wi(h));
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
        s.options[0].replace(Su, function(d, h, p, E, y, B) {
          if (h)
            t.minimumIntegerDigits = p.length;
          else {
            if (E && y)
              throw new Error("We currently do not support maximum integer digits");
            if (B)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (ba.test(s.stem)) {
      t.minimumIntegerDigits = s.stem.length;
      continue;
    }
    if (Si.test(s.stem)) {
      if (s.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      s.stem.replace(Si, function(d, h, p, E, y, B) {
        return p === "*" ? t.minimumFractionDigits = h.length : E && E[0] === "#" ? t.maximumFractionDigits = E.length : y && B ? (t.minimumFractionDigits = y.length, t.maximumFractionDigits = y.length + B.length) : (t.minimumFractionDigits = h.length, t.maximumFractionDigits = h.length), "";
      });
      var u = s.options[0];
      u === "w" ? t = X(X({}, t), { trailingZeroDisplay: "stripIfInteger" }) : u && (t = X(X({}, t), Ti(u)));
      continue;
    }
    if (ga.test(s.stem)) {
      t = X(X({}, t), Ti(s.stem));
      continue;
    }
    var l = ma(s.stem);
    l && (t = X(X({}, t), l));
    var c = Tu(s.stem);
    c && (t = X(X({}, t), c));
  }
  return t;
}
var Er = {
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
function Au(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var s = e.charAt(n);
    if (s === "j") {
      for (var u = 0; n + 1 < e.length && e.charAt(n + 1) === s; )
        u++, n++;
      var l = 1 + (u & 1), c = u < 2 ? 1 : 3 + (u >> 1), d = "a", h = Mu(t);
      for ((h == "H" || h == "k") && (c = 0); c-- > 0; )
        r += d;
      for (; l-- > 0; )
        r = h + r;
    } else s === "J" ? r += "H" : r += s;
  }
  return r;
}
function Mu(e) {
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
  var s = Er[n || ""] || Er[r || ""] || Er["".concat(r, "-001")] || Er["001"];
  return s[0];
}
var an, Ou = new RegExp("^".concat(va.source, "*")), Hu = new RegExp("".concat(va.source, "*$"));
function U(e, t) {
  return { start: e, end: t };
}
var Iu = !!String.prototype.startsWith && "_a".startsWith("a", 1), Nu = !!String.fromCodePoint, Pu = !!Object.fromEntries, Bu = !!String.prototype.codePointAt, Cu = !!String.prototype.trimStart, Lu = !!String.prototype.trimEnd, Ru = !!Number.isSafeInteger, Gu = Ru ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, dn = !0;
try {
  var Du = ya("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  dn = ((an = Du.exec("a")) === null || an === void 0 ? void 0 : an[0]) === "a";
} catch {
  dn = !1;
}
var Ai = Iu ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), pn = Nu ? String.fromCodePoint : (
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
), Mi = (
  // native
  Pu ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, s = t; n < s.length; n++) {
        var u = s[n], l = u[0], c = u[1];
        r[l] = c;
      }
      return r;
    }
  )
), _a = Bu ? (
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
), Uu = Cu ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ou, "");
  }
), Fu = Lu ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Hu, "");
  }
);
function ya(e, t) {
  return new RegExp(e, t);
}
var vn;
if (dn) {
  var Oi = ya("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  vn = function(t, r) {
    var n;
    Oi.lastIndex = r;
    var s = Oi.exec(t);
    return (n = s[1]) !== null && n !== void 0 ? n : "";
  };
} else
  vn = function(t, r) {
    for (var n = []; ; ) {
      var s = _a(t, r);
      if (s === void 0 || xa(s) || Xu(s))
        break;
      n.push(s), r += s >= 65536 ? 2 : 1;
    }
    return pn.apply(void 0, n);
  };
var ku = (
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
            return this.error(D.UNMATCHED_CLOSING_TAG, U(this.clonePosition(), this.clonePosition()));
          } else if (u === 60 && !this.ignoreTag && gn(this.peek() || 0)) {
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
          if (this.isEOF() || !gn(this.char()))
            return this.error(D.INVALID_TAG, U(c, this.clonePosition()));
          var d = this.clonePosition(), h = this.parseTagName();
          return s !== h ? this.error(D.UNMATCHED_CLOSING_TAG, U(d, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: K.tag,
              value: s,
              children: l,
              location: U(n, this.clonePosition())
            },
            err: null
          } : this.error(D.INVALID_TAG, U(c, this.clonePosition())));
        } else
          return this.error(D.UNCLOSED_TAG, U(n, this.clonePosition()));
      } else
        return this.error(D.INVALID_TAG, U(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Vu(this.char()); )
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
      !ju(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return pn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), pn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(D.EMPTY_ARGUMENT, U(n, this.clonePosition()));
      var s = this.parseIdentifierIfPossible().value;
      if (!s)
        return this.error(D.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
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
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition())) : this.parseArgumentOptions(t, r, s, n);
        default:
          return this.error(D.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = vn(this.message, r), s = r + n.length;
      this.bumpTo(s);
      var u = this.clonePosition(), l = U(t, u);
      return { value: n, location: l };
    }, e.prototype.parseArgumentOptions = function(t, r, n, s) {
      var u, l = this.clonePosition(), c = this.parseIdentifierIfPossible().value, d = this.clonePosition();
      switch (c) {
        case "":
          return this.error(D.EXPECT_ARGUMENT_TYPE, U(l, d));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var h = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var p = this.clonePosition(), E = this.parseSimpleArgStyleIfPossible();
            if (E.err)
              return E;
            var y = Fu(E.val);
            if (y.length === 0)
              return this.error(D.EXPECT_ARGUMENT_STYLE, U(this.clonePosition(), this.clonePosition()));
            var B = U(p, this.clonePosition());
            h = { style: y, styleLocation: B };
          }
          var I = this.tryParseArgumentClose(s);
          if (I.err)
            return I;
          var P = U(s, this.clonePosition());
          if (h && Ai(h?.style, "::", 0)) {
            var O = Uu(h.style.slice(2));
            if (c === "number") {
              var E = this.parseNumberSkeletonFromString(O, h.styleLocation);
              return E.err ? E : {
                val: { type: K.number, value: n, location: P, style: E.val },
                err: null
              };
            } else {
              if (O.length === 0)
                return this.error(D.EXPECT_DATE_TIME_SKELETON, P);
              var S = O;
              this.locale && (S = Au(O, this.locale));
              var y = {
                type: kt.dateTime,
                pattern: S,
                location: h.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? _u(S) : {}
              }, M = c === "date" ? K.date : K.time;
              return {
                val: { type: M, value: n, location: P, style: y },
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
          var H = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(D.EXPECT_SELECT_ARGUMENT_OPTIONS, U(H, X({}, H)));
          this.bumpSpace();
          var _ = this.parseIdentifierIfPossible(), b = 0;
          if (c !== "select" && _.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(D.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, U(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var E = this.tryParseDecimalInteger(D.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, D.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (E.err)
              return E;
            this.bumpSpace(), _ = this.parseIdentifierIfPossible(), b = E.val;
          }
          var A = this.tryParsePluralOrSelectOptions(t, c, r, _);
          if (A.err)
            return A;
          var I = this.tryParseArgumentClose(s);
          if (I.err)
            return I;
          var T = U(s, this.clonePosition());
          return c === "select" ? {
            val: {
              type: K.select,
              value: n,
              options: Mi(A.val),
              location: T
            },
            err: null
          } : {
            val: {
              type: K.plural,
              value: n,
              options: Mi(A.val),
              offset: b,
              pluralType: c === "plural" ? "cardinal" : "ordinal",
              location: T
            },
            err: null
          };
        }
        default:
          return this.error(D.INVALID_ARGUMENT_TYPE, U(l, d));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var s = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(D.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, U(s, this.clonePosition()));
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
        n = xu(t);
      } catch {
        return this.error(D.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: kt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? wu(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, s) {
      for (var u, l = !1, c = [], d = /* @__PURE__ */ new Set(), h = s.value, p = s.location; ; ) {
        if (h.length === 0) {
          var E = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var y = this.tryParseDecimalInteger(D.EXPECT_PLURAL_ARGUMENT_SELECTOR, D.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (y.err)
              return y;
            p = U(E, this.clonePosition()), h = this.message.slice(E.offset, this.offset());
          } else
            break;
        }
        if (d.has(h))
          return this.error(r === "select" ? D.DUPLICATE_SELECT_ARGUMENT_SELECTOR : D.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, p);
        h === "other" && (l = !0), this.bumpSpace();
        var B = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? D.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : D.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, U(this.clonePosition(), this.clonePosition()));
        var I = this.parseMessage(t + 1, r, n);
        if (I.err)
          return I;
        var P = this.tryParseArgumentClose(B);
        if (P.err)
          return P;
        c.push([
          h,
          {
            value: I.val,
            location: U(B, this.clonePosition())
          }
        ]), d.add(h), this.bumpSpace(), u = this.parseIdentifierIfPossible(), h = u.value, p = u.location;
      }
      return c.length === 0 ? this.error(r === "select" ? D.EXPECT_SELECT_ARGUMENT_SELECTOR : D.EXPECT_PLURAL_ARGUMENT_SELECTOR, U(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !l ? this.error(D.MISSING_OTHER_CLAUSE, U(this.clonePosition(), this.clonePosition())) : { val: c, err: null };
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
      return u ? (l *= n, Gu(l) ? { val: l, err: null } : this.error(r, d)) : this.error(t, d);
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
      var r = _a(this.message, t);
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
      if (Ai(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && xa(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function gn(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function ju(e) {
  return gn(e) || e === 47;
}
function Vu(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function xa(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Xu(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function bn(e) {
  e.forEach(function(t) {
    if (delete t.location, ca(t) || ha(t))
      for (var r in t.options)
        delete t.options[r].location, bn(t.options[r].value);
    else ua(t) && pa(t.style) || (la(t) || fa(t)) && hn(t.style) ? delete t.style.location : da(t) && bn(t.children);
  });
}
function Wu(e, t) {
  t === void 0 && (t = {}), t = X({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new ku(e, t).parse();
  if (r.err) {
    var n = SyntaxError(D[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || bn(r.val), r.val;
}
var jt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(jt || (jt = {}));
var Rr = (
  /** @class */
  (function(e) {
    Lr(t, e);
    function t(r, n, s) {
      var u = e.call(this, r) || this;
      return u.code = n, u.originalMessage = s, u;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Hi = (
  /** @class */
  (function(e) {
    Lr(t, e);
    function t(r, n, s, u) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(s).join('", "'), '"'), jt.INVALID_VALUE, u) || this;
    }
    return t;
  })(Rr)
), zu = (
  /** @class */
  (function(e) {
    Lr(t, e);
    function t(r, n, s) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), jt.INVALID_VALUE, s) || this;
    }
    return t;
  })(Rr)
), Yu = (
  /** @class */
  (function(e) {
    Lr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), jt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Rr)
), ge;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ge || (ge = {}));
function qu(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ge.literal || r.type !== ge.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Zu(e) {
  return typeof e == "function";
}
function Tr(e, t, r, n, s, u, l) {
  if (e.length === 1 && Ei(e[0]))
    return [
      {
        type: ge.literal,
        value: e[0].value
      }
    ];
  for (var c = [], d = 0, h = e; d < h.length; d++) {
    var p = h[d];
    if (Ei(p)) {
      c.push({
        type: ge.literal,
        value: p.value
      });
      continue;
    }
    if (bu(p)) {
      typeof u == "number" && c.push({
        type: ge.literal,
        value: r.getNumberFormat(t).format(u)
      });
      continue;
    }
    var E = p.value;
    if (!(s && E in s))
      throw new Yu(E, l);
    var y = s[E];
    if (gu(p)) {
      (!y || typeof y == "string" || typeof y == "number") && (y = typeof y == "string" || typeof y == "number" ? String(y) : ""), c.push({
        type: typeof y == "string" ? ge.literal : ge.object,
        value: y
      });
      continue;
    }
    if (la(p)) {
      var B = typeof p.style == "string" ? n.date[p.style] : hn(p.style) ? p.style.parsedOptions : void 0;
      c.push({
        type: ge.literal,
        value: r.getDateTimeFormat(t, B).format(y)
      });
      continue;
    }
    if (fa(p)) {
      var B = typeof p.style == "string" ? n.time[p.style] : hn(p.style) ? p.style.parsedOptions : n.time.medium;
      c.push({
        type: ge.literal,
        value: r.getDateTimeFormat(t, B).format(y)
      });
      continue;
    }
    if (ua(p)) {
      var B = typeof p.style == "string" ? n.number[p.style] : pa(p.style) ? p.style.parsedOptions : void 0;
      B && B.scale && (y = y * (B.scale || 1)), c.push({
        type: ge.literal,
        value: r.getNumberFormat(t, B).format(y)
      });
      continue;
    }
    if (da(p)) {
      var I = p.children, P = p.value, O = s[P];
      if (!Zu(O))
        throw new zu(P, "function", l);
      var S = Tr(I, t, r, n, s, u), M = O(S.map(function(b) {
        return b.value;
      }));
      Array.isArray(M) || (M = [M]), c.push.apply(c, M.map(function(b) {
        return {
          type: typeof b == "string" ? ge.literal : ge.object,
          value: b
        };
      }));
    }
    if (ca(p)) {
      var H = p.options[y] || p.options.other;
      if (!H)
        throw new Hi(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Tr(H.value, t, r, n, s));
      continue;
    }
    if (ha(p)) {
      var H = p.options["=".concat(y)];
      if (!H) {
        if (!Intl.PluralRules)
          throw new Rr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, jt.MISSING_INTL_API, l);
        var _ = r.getPluralRules(t, { type: p.pluralType }).select(y - (p.offset || 0));
        H = p.options[_] || p.options.other;
      }
      if (!H)
        throw new Hi(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Tr(H.value, t, r, n, s, y - (p.offset || 0)));
      continue;
    }
  }
  return qu(c);
}
function Ju(e, t) {
  return t ? X(X(X({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = X(X({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function $u(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Ju(e[n], t[n]), r;
  }, X({}, e)) : e;
}
function sn(e) {
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
function Ku(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: rn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, tn([void 0], r, !1)))();
    }, {
      cache: sn(e.number),
      strategy: nn.variadic
    }),
    getDateTimeFormat: rn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, tn([void 0], r, !1)))();
    }, {
      cache: sn(e.dateTime),
      strategy: nn.variadic
    }),
    getPluralRules: rn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, tn([void 0], r, !1)))();
    }, {
      cache: sn(e.pluralRules),
      strategy: nn.variadic
    })
  };
}
var Qu = (
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
        var p = h.reduce(function(E, y) {
          return !E.length || y.type !== ge.literal || typeof E[E.length - 1] != "string" ? E.push(y.value) : E[E.length - 1] += y.value, E;
        }, []);
        return p.length <= 1 ? p[0] || "" : p;
      }, this.formatToParts = function(d) {
        return Tr(u.ast, u.locales, u.formatters, u.formats, d, void 0, u.message);
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
        var c = uu(l, ["formatters"]);
        this.ast = e.__parse(t, X(X({}, c), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = $u(e.formats, n), this.formatters = s && s.formatters || Ku(this.formatterCache);
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
    }, e.__parse = Wu, e.formats = {
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
function el(e, t) {
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
const pt = {}, tl = (e, t, r) => r && (t in pt || (pt[t] = {}), e in pt[t] || (pt[t][e] = r), r), Ea = (e, t) => {
  if (t == null)
    return;
  if (t in pt && e in pt[t])
    return pt[t][e];
  const r = Gr(t);
  for (let n = 0; n < r.length; n++) {
    const s = r[n], u = nl(s, e);
    if (u)
      return tl(e, t, u);
  }
};
let Mn;
const or = Cr({});
function rl(e) {
  return Mn[e] || null;
}
function Sa(e) {
  return e in Mn;
}
function nl(e, t) {
  if (!Sa(e))
    return null;
  const r = rl(e);
  return el(r, t);
}
function il(e) {
  if (e == null)
    return;
  const t = Gr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Sa(n))
      return n;
  }
}
function al(e, ...t) {
  delete pt[e], or.update((r) => (r[e] = ou.all([r[e] || {}, ...t]), r));
}
Xt(
  [or],
  ([e]) => Object.keys(e)
);
or.subscribe((e) => Mn = e);
const wr = {};
function sl(e, t) {
  wr[e].delete(t), wr[e].size === 0 && delete wr[e];
}
function Ta(e) {
  return wr[e];
}
function ol(e) {
  return Gr(e).map((t) => {
    const r = Ta(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function mn(e) {
  return e == null ? !1 : Gr(e).some(
    (t) => {
      var r;
      return (r = Ta(t)) == null ? void 0 : r.size;
    }
  );
}
function ul(e, t) {
  return Promise.all(
    t.map((n) => (sl(e, n), n().then((s) => s.default || s)))
  ).then((n) => al(e, ...n));
}
const rr = {};
function wa(e) {
  if (!mn(e))
    return e in rr ? rr[e] : Promise.resolve();
  const t = ol(e);
  return rr[e] = Promise.all(
    t.map(
      ([r, n]) => ul(r, n)
    )
  ).then(() => {
    if (mn(e))
      return wa(e);
    delete rr[e];
  }), rr[e];
}
const ll = {
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
}, fl = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: ll,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, cl = fl;
function Vt() {
  return cl;
}
const on = Cr(!1);
var hl = Object.defineProperty, dl = Object.defineProperties, pl = Object.getOwnPropertyDescriptors, Ii = Object.getOwnPropertySymbols, vl = Object.prototype.hasOwnProperty, gl = Object.prototype.propertyIsEnumerable, Ni = (e, t, r) => t in e ? hl(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, bl = (e, t) => {
  for (var r in t || (t = {}))
    vl.call(t, r) && Ni(e, r, t[r]);
  if (Ii)
    for (var r of Ii(t))
      gl.call(t, r) && Ni(e, r, t[r]);
  return e;
}, ml = (e, t) => dl(e, pl(t));
let _n;
const Ir = Cr(null);
function Pi(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Gr(e, t = Vt().fallbackLocale) {
  const r = Pi(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Pi(t)])] : r;
}
function At() {
  return _n ?? void 0;
}
Ir.subscribe((e) => {
  _n = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const _l = (e) => {
  if (e && il(e) && mn(e)) {
    const { loadingDelay: t } = Vt();
    let r;
    return typeof window < "u" && At() != null && t ? r = window.setTimeout(
      () => on.set(!0),
      t
    ) : on.set(!0), wa(e).then(() => {
      Ir.set(e);
    }).finally(() => {
      clearTimeout(r), on.set(!1);
    });
  }
  return Ir.set(e);
}, Wt = ml(bl({}, Ir), {
  set: _l
}), Dr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const s = JSON.stringify(n);
    return s in t ? t[s] : t[s] = e(n);
  };
};
var yl = Object.defineProperty, Nr = Object.getOwnPropertySymbols, Aa = Object.prototype.hasOwnProperty, Ma = Object.prototype.propertyIsEnumerable, Bi = (e, t, r) => t in e ? yl(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, On = (e, t) => {
  for (var r in t || (t = {}))
    Aa.call(t, r) && Bi(e, r, t[r]);
  if (Nr)
    for (var r of Nr(t))
      Ma.call(t, r) && Bi(e, r, t[r]);
  return e;
}, zt = (e, t) => {
  var r = {};
  for (var n in e)
    Aa.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Nr)
    for (var n of Nr(e))
      t.indexOf(n) < 0 && Ma.call(e, n) && (r[n] = e[n]);
  return r;
};
const sr = (e, t) => {
  const { formats: r } = Vt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, xl = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = zt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (s = sr("number", n)), new Intl.NumberFormat(r, s);
  }
), El = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = zt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? s = sr("date", n) : Object.keys(s).length === 0 && (s = sr("date", "short")), new Intl.DateTimeFormat(r, s);
  }
), Sl = Dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = zt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? s = sr("time", n) : Object.keys(s).length === 0 && (s = sr("time", "short")), new Intl.DateTimeFormat(r, s);
  }
), Tl = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = zt(t, [
    "locale"
  ]);
  return xl(On({ locale: r }, n));
}, wl = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = zt(t, [
    "locale"
  ]);
  return El(On({ locale: r }, n));
}, Al = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = zt(t, [
    "locale"
  ]);
  return Sl(On({ locale: r }, n));
}, Ml = Dr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = At()) => new Qu(e, t, Vt().formats, {
    ignoreTag: Vt().ignoreTag
  })
), Ol = (e, t = {}) => {
  var r, n, s, u;
  let l = t;
  typeof e == "object" && (l = e, e = l.id);
  const {
    values: c,
    locale: d = At(),
    default: h
  } = l;
  if (d == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let p = Ea(e, d);
  if (!p)
    p = (u = (s = (n = (r = Vt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: d, id: e, defaultValue: h })) != null ? s : h) != null ? u : e;
  else if (typeof p != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof p}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), p;
  if (!c)
    return p;
  let E = p;
  try {
    E = Ml(p, d).format(c);
  } catch (y) {
    y instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      y.message
    );
  }
  return E;
}, Hl = (e, t) => Al(t).format(e), Il = (e, t) => wl(t).format(e), Nl = (e, t) => Tl(t).format(e), Pl = (e, t = At()) => Ea(e, t);
Xt([Wt, or], () => Ol);
Xt([Wt], () => Hl);
Xt([Wt], () => Il);
Xt([Wt], () => Nl);
Xt([Wt, or], () => Pl);
const Bl = "__i18n__", Cl = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Ll = [
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
function Rl(e) {
  return typeof e == "string" && e.includes(Bl);
}
class Gl {
  load_component;
  #t = Z(dt({}));
  get shared() {
    return a(this.#t);
  }
  set shared(t) {
    m(this.#t, t, !0);
  }
  #r = Z(dt({}));
  get props() {
    return a(this.#r);
  }
  set props(t) {
    m(this.#r, t, !0);
  }
  #e = Z((t) => t);
  get i18n() {
    return a(this.#e);
  }
  set i18n(t) {
    m(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Ll;
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
    for (const n of Cl)
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
    ), Ft(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), ce(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Wt.subscribe(() => {
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
    return So(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], s = Rl(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const u = r;
        this.shared[u] = s;
        continue;
      }
      this.props[r] = s;
    }
  }
  watch_for_change() {
    Ft(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
_o();
var Dl = /* @__PURE__ */ No('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Ci = /* @__PURE__ */ ye("<!> <!>", 1), Ul = /* @__PURE__ */ ye('<div class="placeholder svelte-1stq1b1"></div>');
function Fl(e, t) {
  Ji(t, !1);
  let r = Q(t, "height", 8, void 0), n = Q(t, "min_height", 8, void 0), s = Q(t, "max_height", 8, void 0), u = Q(t, "width", 8, void 0), l = Q(t, "elem_id", 8, ""), c = Q(t, "elem_classes", 24, () => []), d = Q(t, "variant", 8, "solid"), h = Q(t, "border_mode", 8, "base"), p = Q(t, "padding", 8, !0), E = Q(t, "type", 8, "normal"), y = Q(t, "test_id", 8, void 0), B = Q(t, "explicit_call", 8, !1), I = Q(t, "container", 8, !0), P = Q(t, "visible", 8, !0), O = Q(t, "allow_overflow", 8, !0), S = Q(t, "overflow_behavior", 8, "auto"), M = Q(t, "scale", 8, null), H = Q(t, "min_width", 8, 0), _ = Q(t, "flex", 12, !1), b = Q(t, "resizable", 8, !1), A = Q(t, "rtl", 8, !1), T = Q(t, "fullscreen", 12, !1), F = Q(t, "label", 8, void 0), ee = Lt(T()), G = Lt(), vt = E() === "fieldset" ? "fieldset" : "div", Ce = Lt(0), tt = Lt(0), ae = Lt(null);
  function Fe(te) {
    T() && te.key === "Escape" && T(!1);
  }
  const Je = (te) => {
    if (te !== void 0) {
      if (typeof te == "number")
        return te + "px";
      if (typeof te == "string")
        return te;
    }
  }, be = (te) => {
    let Oe = te.clientY;
    const xe = (rt) => {
      const Ee = rt.clientY - Oe;
      Oe = rt.clientY, xo(G, a(G).style.height = `${a(G).offsetHeight + Ee}px`);
    }, $e = () => {
      window.removeEventListener("mousemove", xe), window.removeEventListener("mouseup", $e);
    };
    window.addEventListener("mousemove", xe), window.addEventListener("mouseup", $e);
  };
  hi(
    () => (Be(T()), a(ee), a(G)),
    () => {
      T() !== a(ee) && (m(ee, T()), T() ? (m(ae, a(G).getBoundingClientRect()), m(Ce, a(G).offsetHeight), m(tt, a(G).offsetWidth), window.addEventListener("keydown", Fe)) : (m(ae, null), window.removeEventListener("keydown", Fe)));
    }
  ), hi(() => Be(P()), () => {
    P() || _(!1);
  }), yo(), eu();
  var ke = Po(), gt = Jr(ke);
  {
    var Le = (te) => {
      var Oe = Ci(), xe = Jr(Oe);
      Fo(xe, () => vt, !1, (Ee, Re) => {
        aa(Ee, (Ge) => m(G, Ge), () => a(G)), Qo(
          Ee,
          (Ge, De) => ({
            "data-testid": y(),
            id: l(),
            class: `block ${Ge ?? ""}`,
            dir: A() ? "rtl" : "ltr",
            "aria-label": F(),
            style: "",
            [ir]: {
              hidden: P() === "hidden",
              padded: p(),
              flex: _(),
              border_focus: h() === "focus",
              border_contrast: h() === "contrast",
              "hide-container": !B() && !I(),
              fullscreen: T(),
              animating: T() && a(ae) !== null,
              "auto-margin": M() === null
            },
            [Rt]: De
          }),
          [
            () => (Be(c()), ce(() => c()?.join(" ") || "")),
            () => ({
              height: (Be(T()), Be(r()), ce(() => T() ? void 0 : Je(r()))),
              "min-height": (Be(T()), Be(n()), ce(() => T() ? void 0 : Je(n()))),
              "max-height": (Be(T()), Be(s()), ce(() => T() ? void 0 : Je(s()))),
              "--start-top": (a(ae), ce(() => a(ae) ? `${a(ae).top}px` : "0px")),
              "--start-left": (a(ae), ce(() => a(ae) ? `${a(ae).left}px` : "0px")),
              "--start-width": (a(ae), ce(() => a(ae) ? `${a(ae).width}px` : "0px")),
              "--start-height": (a(ae), ce(() => a(ae) ? `${a(ae).height}px` : "0px")),
              width: (Be(T()), Be(u()), ce(() => T() ? void 0 : typeof u() == "number" ? `calc(min(${u()}px, 100%))` : Je(u()))),
              "border-style": d(),
              overflow: O() ? S() : "hidden",
              "flex-grow": M(),
              "min-width": `calc(min(${H()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var He = Ci(), nt = Jr(He);
        Uo(nt, t, "default", {});
        var it = Y(nt, 2);
        {
          var Yt = (Ge) => {
            var De = Dl();
            V("mousedown", De, be), le(Ge, De);
          };
          qe(it, (Ge) => {
            b() && Ge(Yt);
          });
        }
        le(Re, He);
      });
      var $e = Y(xe, 2);
      {
        var rt = (Ee) => {
          var Re = Ul();
          let He;
          Ye(() => He = Or(Re, "", He, {
            height: a(Ce) + "px",
            width: a(tt) + "px"
          })), le(Ee, Re);
        };
        qe($e, (Ee) => {
          T() && Ee(rt);
        });
      }
      le(te, Oe);
    };
    qe(gt, (te) => {
      (P() === !0 || P() === "hidden") && te(Le);
    });
  }
  le(e, ke), Zi();
}
wn(["click"]);
wn(["click"]);
var kl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), jl = /* @__PURE__ */ ye("<option> </option>"), Vl = /* @__PURE__ */ ye('<span class="current-badge svelte-r41nsf">当前</span>'), Xl = /* @__PURE__ */ ye('<div role="button" tabindex="0"><span class="group-drag-handle svelte-r41nsf" title="拖动调整叠层顺序" aria-label="拖动调整叠层顺序">⠿</span> <span class="group-order svelte-r41nsf"></span> <span class="group-reorder-label svelte-r41nsf"> <!></span> <button type="button" class="visibility-button svelte-r41nsf"> </button> <button type="button" title="只显示此层">S</button></div>'), Wl = /* @__PURE__ */ ye('<div class="group-reorder svelte-r41nsf" aria-label="调整 Label 叠层顺序"><div class="group-reorder-hint svelte-r41nsf">拖动排序；眼睛与单层显示仅影响浏览器预览</div> <div class="group-reorder-list svelte-r41nsf"></div></div>'), zl = /* @__PURE__ */ ye('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select> <div class="mode-switch svelte-r41nsf" aria-label="显示模式"><button type="button">对齐模式</button> <button type="button">交叠预览</button></div> <!></div>'), Yl = /* @__PURE__ */ ye('<label class="svelte-r41nsf"><span> </span> <input type="number" class="svelte-r41nsf"/> <small class="svelte-r41nsf"> </small></label>'), ql = /* @__PURE__ */ ye('<div class="numeric-toolbar svelte-r41nsf"></div>'), Zl = /* @__PURE__ */ ye('<button type="button" class="svelte-r41nsf"> </button>'), Jl = /* @__PURE__ */ ye('<div class="proposal-actions svelte-r41nsf"></div>'), $l = /* @__PURE__ */ ye('<div><div class="proposal-copy svelte-r41nsf"><strong class="svelte-r41nsf"> </strong> <span> </span></div> <!></div>'), Kl = /* @__PURE__ */ ye('<div class="overlap-legend svelte-r41nsf"><span class="legend-current svelte-r41nsf"> </span> <span class="legend-next svelte-r41nsf"> </span> <span class="legend-overlap svelte-r41nsf"> </span></div>'), Ql = /* @__PURE__ */ ye('<div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回标注</button> <button type="button" class="svelte-r41nsf">画布 −</button> <button type="button" class="svelte-r41nsf"> </button> <button type="button" class="svelte-r41nsf">画布 ＋</button></div> <!> <!> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <!> <div class="status svelte-r41nsf"><span> </span> <span class="svelte-r41nsf"> </span></div></div>');
function tf(e, t) {
  Ji(t, !0);
  const r = /* @__PURE__ */ ru(t, kl), n = 2048, s = Math.log(1.01) / 100, u = 250, l = 1e4, c = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], d = new Gl(r);
  let h, p = null, E = null, y = null, B = null, I = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), O = Z(0), S = null, M = Z(!1), H = Z(!1), _ = Z("等待版图 mask"), b = Z(dt({ enabled: !1 })), A = Z(dt(Hn())), T = Z(""), F = Z(!1), ee = Z(!1), G = Z("crosshair"), vt = "", Ce = 0, tt = { x: 0, y: 0, center_x: 0, center_y: 0 }, ae = { angle: 0, rotation: 0 }, Fe = null, Je = null, be = null, ke = null, gt = null, Le = 0, te = 0, Oe = "", xe = "", $e = "", rt = /* @__PURE__ */ new Set(), Ee = Z(""), Re = Z(""), He = Z("before"), nt = Z(!1), it = Z("align"), Yt = Z(dt(/* @__PURE__ */ new Set())), Ge = Z(dt(/* @__PURE__ */ new Set())), De = Z(""), he = Z(null), Ur = /* @__PURE__ */ new Set(), W = Z(1), je = Z(0), Ve = Z(0), Mt = Z(!1), ur = Z(!1), qt = { client_x: 0, client_y: 0, pan_x: 0, pan_y: 0 }, Zt = null, lr = Z(""), at = Z(dt({ x: "0", y: "0", scale: "100", rotation: "0", alpha: "22" }));
  function Hn() {
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
  function In(i) {
    return JSON.parse(JSON.stringify(i || { enabled: !1 }));
  }
  function Se(i) {
    const o = Object.assign(Object.assign({}, Hn()), i || {});
    return Object.assign(Object.assign({}, o), {
      center_x: ne(o.center_x, 0),
      center_y: ne(o.center_y, 0),
      pivot_x: ne(o.pivot_x, 0),
      pivot_y: ne(o.pivot_y, 0),
      scale: Te(ne(o.scale, 1), 0.01, 20),
      rotation_deg: fr(ne(o.rotation_deg, 0)),
      preview_alpha: Te(ne(o.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(ne(o.revision, 0)))
    });
  }
  function ne(i, o) {
    const f = Number(i);
    return Number.isFinite(f) ? f : o;
  }
  function Oa(i) {
    return typeof i == "number" ? String(i) + "px" : i || "520px";
  }
  function Te(i, o, f) {
    return Math.max(o, Math.min(f, i));
  }
  function fr(i) {
    let o = ((i + 180) % 360 + 360) % 360 - 180;
    return o === -180 && (o = 180), o;
  }
  function k(i) {
    return typeof i + ":" + String(i);
  }
  function Xe() {
    var i;
    return z() ? Array.isArray((i = a(b).group_view) === null || i === void 0 ? void 0 : i.groups) ? a(b).group_view.groups : [] : [];
  }
  function Ot() {
    var i;
    return z() && ((i = a(b).group_view) === null || i === void 0 ? void 0 : i.allow_reorder) === !0;
  }
  function Ht() {
    var i;
    return z() && ((i = a(b).group_view) === null || i === void 0 ? void 0 : i.same_layer) === !0;
  }
  function z() {
    return a(b).transform_mode === "label_groups" && !!a(b).group_view;
  }
  function Nn() {
    for (const i of Xe())
      if (k(i.group_id) === a(T)) return i;
    return null;
  }
  function de() {
    const i = Nn();
    return i ? String(i.label || "Label " + String(i.group_id)) : "";
  }
  function Jt() {
    const i = Xe(), o = i.findIndex((f) => k(f.group_id) === a(T));
    return o < 0 || o + 1 >= i.length ? "" : k(i[o + 1].group_id);
  }
  function Ha(i) {
    return a(De) ? i === a(De) : i === a(T) ? !0 : a(it) === "overlap" && i === Jt();
  }
  function cr(i) {
    return Ht() ? !0 : a(Yt).has(i) ? !1 : a(Ge).has(i) ? !0 : Ha(i);
  }
  function Pn() {
    return Xe().map((i) => k(i.group_id)).filter((i) => cr(i));
  }
  function fe() {
    return Math.max(1, Number(a(b).target_width || p?.naturalWidth || 1));
  }
  function we() {
    return Math.max(1, Number(a(b).target_height || p?.naturalHeight || 1));
  }
  function Ia() {
    var i;
    const o = (i = I.get(a(T))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(b).source_width || E?.naturalWidth || o?.naturalWidth || 1));
  }
  function Na() {
    var i;
    const o = (i = I.get(a(T))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(b).source_height || E?.naturalHeight || o?.naturalHeight || 1));
  }
  function Pa() {
    return Math.min(1, n / Math.max(fe(), we()));
  }
  function Bn(i = a(T)) {
    var o;
    if (z()) {
      const v = (o = I.get(i)) === null || o === void 0 ? void 0 : o.view.foreground_bbox_xyxy;
      if (Array.isArray(v) && v.length >= 4) return v.slice(0, 4).map(Number);
    }
    const f = a(b).foreground_bbox_xyxy;
    return Array.isArray(f) && f.length >= 4 ? f.slice(0, 4).map(Number) : [0, 0, Ia() - 1, Na() - 1];
  }
  function Fr(i, o, f) {
    if (!i) {
      o === Ce && f(null);
      return;
    }
    const v = new Image();
    v.onload = () => {
      o === Ce && f(v);
    }, v.onerror = () => {
      o === Ce && f(null);
    }, v.src = i;
  }
  function Cn(i, o) {
    const f = Math.max(1, Number(a(b).source_width || i.naturalWidth || 1)), v = Math.max(1, Number(a(b).source_height || i.naturalHeight || 1)), g = document.createElement("canvas");
    g.width = f, g.height = v;
    const x = g.getContext("2d", { willReadFrequently: !0 });
    if (!x) return null;
    x.imageSmoothingEnabled = !1, x.drawImage(i, 0, 0, f, v);
    const w = x.getImageData(0, 0, f, v), N = document.createElement("canvas");
    N.width = f, N.height = v;
    const C = N.getContext("2d");
    if (!C) return null;
    const L = C.createImageData(f, v);
    for (let R = 0; R < w.data.length; R += 4) {
      const J = Math.max(w.data[R], w.data[R + 1], w.data[R + 2]);
      w.data[R + 3] > 0 && J >= 128 && (L.data[R] = o[0], L.data[R + 1] = o[1], L.data[R + 2] = o[2], L.data[R + 3] = 255);
    }
    return C.putImageData(L, 0, 0), { mask: g, tint: N };
  }
  function Ba() {
    if (!E) {
      y = null, B = null;
      return;
    }
    const i = Cn(E, c[0]);
    y = i?.mask || null, B = i?.tint || null;
  }
  function Ca(i) {
    if (!i.image) {
      i.maskCanvas = null, i.tintCanvas = null, i.ready = !1;
      return;
    }
    const o = Cn(i.image, i.color);
    i.maskCanvas = o?.mask || null, i.tintCanvas = o?.tint || null, i.ready = !!o;
  }
  function It() {
    Fe && (clearTimeout(Fe), Fe = null);
  }
  function Nt() {
    Je && (clearTimeout(Je), Je = null);
  }
  function La(i) {
    const o = JSON.stringify(i);
    let f = 2166136261;
    for (let v = 0; v < o.length; v++)
      f ^= o.charCodeAt(v), f = Math.imul(f, 16777619);
    return "fnv1a32:" + (f >>> 0).toString(16).padStart(8, "0");
  }
  function pe() {
    Zt === null && (Zt = requestAnimationFrame(() => {
      Zt = null, Bt();
    }));
  }
  function Pt() {
    a(lr) || m(
      at,
      {
        x: ne(a(A).center_x, 0).toFixed(2).replace(/\.00$/, ""),
        y: ne(a(A).center_y, 0).toFixed(2).replace(/\.00$/, ""),
        scale: (ne(a(A).scale, 1) * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        rotation: ne(a(A).rotation_deg, 0).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        alpha: (ne(a(A).preview_alpha, 0.22) * 100).toFixed(1).replace(/\.0$/, "")
      },
      !0
    );
  }
  function Ra(i, o) {
    if (!Array.isArray(o) || o.length !== i.length) return i;
    const f = /* @__PURE__ */ new Map();
    for (const x of i) {
      if (!x || x.group_id === void 0 || x.group_id === null) return i;
      const w = k(x.group_id);
      if (f.has(w)) return i;
      f.set(w, x);
    }
    const v = [], g = /* @__PURE__ */ new Set();
    for (const x of o) {
      if (typeof x != "string" && typeof x != "number") return i;
      const w = k(x), N = f.get(w);
      if (!N || g.has(w)) return i;
      g.add(w), v.push(N);
    }
    return v.length === i.length ? v : i;
  }
  function Ga(i) {
    var o;
    const f = a(b).group_view, v = Array.isArray(f?.groups) ? f.groups : [], g = a(b).group_intent, x = !!f && !!g && String(g.selection_signature || "") === String(f.selection_signature || "");
    Oe = String(a(b).editor_session_id || g?.editor_session_id || ""), xe = String(a(b).document_epoch || g?.document_epoch || f?.selection_signature || ""), Le = Math.max(0, Math.trunc(ne(
      (o = g?.client_revision) !== null && o !== void 0 ? o : g?.transform_set_revision,
      0
    ))), te = Le, be = null, ke = null, Nt();
    const w = f?.allow_reorder === !0 && x ? Ra(v, g?.ordered_group_ids) : v;
    w !== v && f && m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        group_view: Object.assign(Object.assign({}, f), { groups: w })
      }),
      !0
    );
    const N = /* @__PURE__ */ new Map();
    if (x && Array.isArray(g?.transforms))
      for (const j of g.transforms)
        !j || j.group_id === void 0 || !j.transform || N.set(k(j.group_id), Se(j.transform));
    const C = Se(a(b).transform), L = /* @__PURE__ */ new Map(), R = /* @__PURE__ */ new Map(), J = /* @__PURE__ */ new Set();
    for (let j = 0; j < w.length; j++) {
      const Ie = w[j];
      if (!Ie || Ie.group_id === void 0 || Ie.group_id === null) continue;
      const Ke = k(Ie.group_id);
      if (J.has(Ke)) continue;
      J.add(Ke);
      const Qe = {
        view: Ie,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: c[j % c.length]
      };
      L.set(Ke, Qe), R.set(Ke, Se(N.get(Ke) || C));
    }
    I = L, P = R, rt = /* @__PURE__ */ new Set(), Ur = /* @__PURE__ */ new Set(), m(he, null);
    const q = x && g?.active_group_id !== void 0 && g.active_group_id !== null ? k(g.active_group_id) : "";
    m(
      T,
      L.has(q) ? q : L.keys().next().value || "",
      !0
    ), m(A, Se(P.get(a(T)) || C), !0), a(T) && P.set(a(T), a(A)), Pt(), E = null, y = null, B = null, m(H, !1);
    for (const [j, Ie] of L)
      Fr(Ie.view.mask_image, i, (Ke) => {
        const Qe = I.get(j);
        !Qe || Qe !== Ie || (Qe.image = Ke, Ca(Qe), m(O, a(O) + 1), pe());
      });
    m(
      _,
      a(b).status || (w.length > 0 ? "已加载 " + String(w.length) + " 个 Label；当前：" + de() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function Da(i) {
    It(), Nt(), be = null, ke = null, gt = null, Ce += 1;
    const o = Ce;
    m(b, In(i), !0), m(M, !1), I = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), m(T, ""), m(F, !1), m(ee, !1), Fr(a(b).base_image, o, (f) => {
      p = f, m(M, !!f), Bt();
    }), z() ? Ga(o) : (m(A, Se(a(b).transform), !0), Pt(), m(_, a(b).status || "编辑器已加载", !0), m(H, !1), Fr(a(b).mask_image, o, (f) => {
      E = f, m(H, !!f), Ba(), Bt();
    }));
  }
  function Ln() {
    return Array.isArray(a(he) === null || a(he) === void 0 ? void 0 : a(he).candidates) ? a(he).candidates : [];
  }
  function Rn(i, o = !1) {
    if (!a(he) || a(he).target_layer_id === void 0) return;
    const f = k(a(he).target_layer_id);
    if (!P.has(f)) return;
    if (o && (Ur.has(f) || ne(We(f).revision, 0) !== ne(a(he).start_transform_revision, 0))) {
      m(_, "粗对齐结果已到达，但检测到手动修改，未覆盖当前变换"), st();
      return;
    }
    Qt(f);
    const v = Se(We(f));
    ze(Object.assign(Object.assign(Object.assign({}, v), i.transform), { preview_alpha: v.preview_alpha }), !1), m(he, null), vr("boundary_alignment", (o ? "已自动应用 " : "已应用 ") + String(i.candidate_id || "候选") + "；停止 250ms 后同步");
  }
  function Ua(i) {
    const o = String(i.editor_session_id || ""), f = String(i.document_epoch || "");
    if (o && o !== Oe || f && f !== xe) return;
    m(he, In(i), !0), m(_, String(i.status || "粗对齐候选已生成"), !0), st();
    const v = Array.isArray(i.candidates) ? i.candidates : [];
    i.auto_apply && v.length ? Rn(v[0], !0) : pe();
  }
  function Fa(i) {
    var o;
    const f = String(i.editor_session_id || ""), v = String(i.document_epoch || "");
    if (f !== Oe || v !== xe)
      return;
    if (i.payload_kind === "ack_error") {
      Nt(), be = null, ke = null, m(_, String(i.status || "同步失败；本地修改仍保留"), !0), st(), Bt();
      return;
    }
    const g = Math.max(0, Math.trunc(ne(i.server_ack_revision, 0))), x = String(i.payload_hash || "");
    be !== null && (g !== be || x !== ke) || be === null && (g !== Le || x !== String(((o = a(b).group_intent) === null || o === void 0 ? void 0 : o.payload_hash) || "")) || (Nt(), te = Math.max(te, g), gt = null, be = null, ke = null, m(_, Le > te ? "本地修改待同步" : "已同步", !0), st(), Le > te && dr(0), Bt());
  }
  Ft(() => {
    const i = d.props.value || null, o = JSON.stringify(i);
    if (i?.payload_kind === "ack" || i?.payload_kind === "ack_error") {
      o !== $e && ($e = o, Fa(i));
      return;
    }
    if (i?.payload_kind === "alignment_proposal") {
      o !== $e && ($e = o, Ua(i));
      return;
    }
    o !== vt && (vt = o, Da(i));
  }), Co(() => {
    Ce += 1, It(), Nt(), Zt !== null && cancelAnimationFrame(Zt);
  });
  function kr(i) {
    const o = Number(i.rotation_deg || 0) * Math.PI / 180, f = Number(i.scale || 1), v = Math.cos(o), g = Math.sin(o), x = f * v, w = f * g, N = Number(i.center_x || 0) - x * Number(i.pivot_x || 0) + w * Number(i.pivot_y || 0), C = Number(i.center_y || 0) - w * Number(i.pivot_x || 0) - x * Number(i.pivot_y || 0);
    return [x, w, -w, x, N, C];
  }
  function $t(i, o, f = a(A)) {
    const [v, g, x, w, N, C] = kr(f);
    return { x: v * i + x * o + N, y: g * i + w * o + C };
  }
  function jr(i, o, f = a(A)) {
    const [v, g, x, w, N, C] = kr(f), L = v * w - g * x;
    if (Math.abs(L) < 1e-9) return { x: -1, y: -1 };
    const R = i - N, J = o - C;
    return { x: (w * R - x * J) / L, y: (-g * R + v * J) / L };
  }
  function Kt(i) {
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * fe(), v = (i.clientY - o.top) / Math.max(1, o.height) * we();
    return {
      x: (f - a(je)) / Math.max(0.05, a(W)),
      y: (v - a(Ve)) / Math.max(0.05, a(W))
    };
  }
  function Vr(i, o) {
    i.setTransform(o * a(W), 0, 0, o * a(W), o * a(je), o * a(Ve));
  }
  function Gn(i, o, f) {
    if (!i) return !1;
    const v = Math.round(o), g = Math.round(f);
    if (v < 0 || g < 0 || v >= i.width || g >= i.height) return !1;
    const x = i.getContext("2d", { willReadFrequently: !0 });
    if (!x) return !1;
    const w = x.getImageData(v, g, 1, 1).data;
    return w[3] > 0 && Math.max(w[0], w[1], w[2]) >= 128;
  }
  function We(i) {
    return !z() || i === a(T) ? a(A) : P.get(i) || a(A);
  }
  function ze(i, o = !0) {
    m(A, Se(i), !0), z() && a(T) && (P.set(a(T), a(A)), o && Ur.add(a(T))), Pt();
  }
  function Dn(i, o) {
    const f = jr(i, o, a(A));
    return Gn(y, f.x, f.y);
  }
  function Un(i, o, f) {
    const v = I.get(i);
    if (!v?.ready) return !1;
    const g = jr(o, f, We(i));
    return Gn(v.maskCanvas, g.x, g.y);
  }
  function Xr(i, o) {
    const f = new Set(Pn());
    if (a(T) && f.has(a(T)) && Un(a(T), i, o)) return a(T);
    const v = Xe().map((g) => k(g.group_id)).reverse();
    for (const g of v)
      if (g !== a(T) && f.has(g) && Un(g, i, o)) return g;
    return "";
  }
  function hr(i = a(T), o = We(i)) {
    const [f, v, g, x] = Bn(i);
    return [
      $t(f, v, o),
      $t(g, v, o),
      $t(g, x, o),
      $t(f, x, o)
    ];
  }
  function Fn() {
    const i = hr();
    let o = i[0];
    for (const w of i)
      (w.y < o.y || Math.abs(w.y - o.y) < 1e-6 && w.x > o.x) && (o = w);
    const f = Number(a(A).rotation_deg || 0) * Math.PI / 180, v = Math.cos(f - Math.PI / 4), g = Math.sin(f - Math.PI / 4), x = 34 / Math.max(0.05, a(W));
    return { x: o.x + v * x, y: o.y + g * x };
  }
  function kn() {
    const i = h?.getBoundingClientRect();
    return i ? Math.max(8 / a(W), 14 * fe() / Math.max(1, i.width) / a(W)) : 14 / a(W);
  }
  function jn(i, o) {
    if (z() && !a(T)) return !1;
    const f = Fn(), v = kn();
    return Math.hypot(i - f.x, o - f.y) <= v;
  }
  function Vn(i) {
    const o = z() ? a(T) : "";
    ze(Object.assign(Object.assign({}, a(A)), {
      revision: Number(a(A).revision || 0) + 1,
      origin: i,
      scale: Te(Number(a(A).scale || 1), 0.01, 20),
      rotation_deg: fr(Number(a(A).rotation_deg || 0))
    })), o && rt.add(o);
  }
  function ka() {
    const i = [];
    for (const o of Xe()) {
      const f = k(o.group_id);
      i.push({
        group_id: o.group_id,
        transform: Object.assign({}, Se(We(f)))
      });
    }
    return i;
  }
  function st() {
    var i;
    const o = {
      payload_kind: "intent",
      editor_session_id: Oe,
      document_epoch: xe,
      payload_hash: String(((i = a(b).group_intent) === null || i === void 0 ? void 0 : i.payload_hash) || ""),
      enabled: a(b).enabled,
      transform: Object.assign({}, Se(a(A))),
      target_width: a(b).target_width,
      target_height: a(b).target_height
    };
    z() && (o.transform_mode = "label_groups", o.group_intent = a(b).group_intent ? JSON.parse(JSON.stringify(a(b).group_intent)) : null), d.props.value = o, vt = JSON.stringify(o);
  }
  function ja() {
    if (!z()) {
      d.dispatch("change");
      return;
    }
    const i = a(b).group_intent, o = Math.max(0, Math.trunc(ne(i?.client_revision, 0)));
    !i || o <= te || be !== null || (Nt(), be = o, ke = String(i.payload_hash || ""), m(_, "同步中…"), st(), d.dispatch("change"), Je = setTimeout(
      () => {
        if (be !== o) return;
        be = null, ke = null;
        const f = gt !== o;
        gt = o, m(_, f ? "同步超时，正在重试；本地修改仍保留" : "同步失败；可继续编辑或直接保存重试", !0), st(), Bt(), f && dr(1e3);
      },
      l
    ));
  }
  function dr(i = u) {
    It(), Fe = setTimeout(
      () => {
        Fe = null, ja();
      },
      Math.max(0, i)
    );
  }
  function pr(i, o, f, v = !0) {
    var g;
    if (!z()) return;
    f && Vn(i);
    const x = a(b).group_view, w = Nn(), N = Xe().filter((R) => rt.has(k(R.group_id))).map((R) => R.group_id), C = Le + 1, L = Object.assign(
      Object.assign(
        {
          selection_signature: String(x?.selection_signature || xe),
          transform_set_revision: C,
          editor_session_id: Oe,
          document_epoch: xe,
          client_revision: C,
          payload_hash: "",
          active_group_id: (g = w?.group_id) !== null && g !== void 0 ? g : null,
          changed_group_ids: N
        },
        Ot() ? {
          ordered_group_ids: Xe().map((R) => R.group_id)
        } : {}
      ),
      { transforms: ka() }
    );
    L.payload_hash = La(Object.assign(Object.assign({}, L), { payload_hash: "" })), Le = C, gt = null, m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        payload_kind: "intent",
        editor_session_id: Oe,
        document_epoch: xe,
        payload_hash: L.payload_hash,
        group_intent: L
      }),
      !0
    ), m(_, o + "（本地修改）"), st(), v && dr(0), pe();
  }
  function bt(i, o, f = !0) {
    if (z()) {
      pr(i, o, !0, f);
      return;
    }
    Vn(i), m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        enabled: !0,
        transform: Object.assign({}, a(A)),
        status: o
      }),
      !0
    ), m(_, o, !0), st(), f && (It(), d.dispatch("change")), pe();
  }
  function vr(i, o, f = u) {
    if (z()) {
      bt(i, o, !1), dr(f);
      return;
    }
    bt(i, o, !1), It(), Fe = setTimeout(
      () => {
        Fe = null, d.dispatch("change");
      },
      f
    );
  }
  function Va(i, o, f) {
    if (p && a(M)) {
      i.imageSmoothingEnabled = !0, i.drawImage(p, 0, 0, o, f);
      return;
    }
    i.fillStyle = "#f8fafc", i.fillRect(0, 0, o, f), i.fillStyle = "#64748b", i.font = "18px sans-serif", i.fillText("请先上传图像", 24, 42);
  }
  function gr(i, o, f, v, g) {
    var x;
    g === void 0 && (g = Te(Number((x = v.preview_alpha) !== null && x !== void 0 ? x : 0.35), 0, 1)), i.save(), i.globalAlpha = g;
    const [w, N, C, L, R, J] = kr(v);
    i.setTransform(o * a(W) * w, o * a(W) * N, o * a(W) * C, o * a(W) * L, o * (a(W) * R + a(je)), o * (a(W) * J + a(Ve))), i.imageSmoothingEnabled = !1, i.drawImage(f, 0, 0, f.width, f.height), i.restore();
  }
  function Xa(i, o, f, v) {
    const g = I.get(f), x = I.get(v);
    if (!g?.tintCanvas || !x?.tintCanvas || !g.ready || !x.ready) return;
    const w = S || document.createElement("canvas");
    S = w, w.width !== h.width && (w.width = h.width), w.height !== h.height && (w.height = h.height);
    const N = w.getContext("2d");
    if (N) {
      N.setTransform(1, 0, 0, 1, 0, 0), N.globalCompositeOperation = "source-over", N.clearRect(0, 0, w.width, w.height), gr(N, o, g.tintCanvas, We(f), 1), N.globalCompositeOperation = "destination-in", gr(N, o, x.tintCanvas, We(v), 1), N.globalCompositeOperation = "source-in", N.setTransform(1, 0, 0, 1, 0, 0), N.fillStyle = "#ff3fa4", N.fillRect(0, 0, w.width, w.height), N.globalCompositeOperation = "source-atop", N.strokeStyle = "rgba(255,255,255,0.78)", N.lineWidth = 1.5;
      for (let C = -w.height; C < w.width; C += 14)
        N.beginPath(), N.moveTo(C, 0), N.lineTo(C + w.height, w.height), N.stroke();
      i.save(), i.setTransform(1, 0, 0, 1, 0, 0), i.globalAlpha = 0.62, i.drawImage(w, 0, 0), i.restore();
    }
  }
  function Xn(i, o) {
    const f = hr();
    i.save(), i.lineJoin = "round", i.lineWidth = Math.max(2.5, o / 700) / a(W), i.strokeStyle = "rgba(0,0,0,0.82)", i.beginPath(), i.moveTo(f[0].x, f[0].y);
    for (let x = 1; x < f.length; x++) i.lineTo(f[x].x, f[x].y);
    i.closePath(), i.stroke(), i.lineWidth = Math.max(1.8, o / 1e3) / a(W), i.strokeStyle = "#00ff66", i.stroke();
    const v = Fn(), g = f.reduce((x, w) => w.y < x.y || Math.abs(w.y - x.y) < 1e-6 && w.x > x.x ? w : x, f[0]);
    i.strokeStyle = "#0f172a", i.lineWidth = Math.max(2, o / 900) / a(W), i.beginPath(), i.moveTo(g.x, g.y), i.lineTo(v.x, v.y), i.stroke(), i.fillStyle = a(ee) ? "#ffb000" : "#ffffff", i.strokeStyle = "#00ff66", i.lineWidth = Math.max(2, o / 900) / a(W), i.beginPath(), i.arc(v.x, v.y, kn(), 0, Math.PI * 2), i.fill(), i.stroke(), i.restore();
  }
  function Wa(i, o, f) {
    const v = I.get(o);
    if (!v) return;
    const g = hr(o, We(o)), x = Math.min(...g.map((j) => j.x)), w = Math.min(...g.map((j) => j.y)), N = String(v.view.label || "Label " + String(v.view.group_id)), L = fe() / Math.max(1, h.getBoundingClientRect().width) / Math.max(0.05, a(W));
    i.save(), i.font = "600 " + String(13 * L) + "px sans-serif";
    const R = i.measureText(N).width + 12 * L, J = Te(x, 2, Math.max(2, fe() - R - 2)), q = Te(w - 23 * L, 2, Math.max(2, we() - 22 * L));
    i.fillStyle = f ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.62)", i.fillRect(J, q, R, 20 * L), i.fillStyle = "#ffffff", i.fillText(N, J + 6 * L, q + 14 * L), i.restore();
  }
  function Bt() {
    var i, o, f;
    if (!h) return;
    const v = fe(), g = we(), x = Pa(), w = Math.max(1, Math.round(v * x)), N = Math.max(1, Math.round(g * x));
    h.width !== w && (h.width = w), h.height !== N && (h.height = N);
    const C = h.getContext("2d");
    if (C && (C.setTransform(1, 0, 0, 1, 0, 0), C.clearRect(0, 0, h.width, h.height), Vr(C, x), Va(C, v, g), a(b).enabled !== !1)) {
      if (z()) {
        const L = Pn(), R = L.filter((q) => q !== a(T));
        L.includes(a(T)) && R.push(a(T));
        const J = Jt();
        for (const q of R) {
          const j = I.get(q);
          if (j?.ready && j.tintCanvas) {
            const Ie = Ht() || q === a(T) ? Te(Number((i = We(q).preview_alpha) !== null && i !== void 0 ? i : 0.22), 0, 1) : q === J ? 0.1 : 0.08;
            gr(C, x, j.tintCanvas, We(q), Ie);
          }
        }
        !Ht() && a(it) === "overlap" && a(T) && J && L.includes(a(T)) && L.includes(J) && Xa(C, x, a(T), J), Vr(C, x);
        for (const q of R)
          !((o = I.get(q)) === null || o === void 0) && o.ready && Wa(C, q, q === a(T));
        a(T) && L.includes(a(T)) && (!((f = I.get(a(T))) === null || f === void 0) && f.ready) && Xn(C, v);
        return;
      }
      B && a(H) && (gr(C, x, B, a(A)), Vr(C, x), Xn(C, v));
    }
  }
  function me() {
    var i;
    return a(b).enabled ? z() ? (a(O), !!a(T) && !!(!((i = I.get(a(T))) === null || i === void 0) && i.ready)) : a(H) && !!y : !1;
  }
  function Qt(i) {
    return !z() || !I.has(i) || i === a(T) ? !1 : (a(T) && P.set(a(T), Se(a(A))), m(T, i, !0), m(A, Se(P.get(i)), !0), P.set(i, a(A)), Pt(), m(_, "当前 Label：" + de()), pe(), !0);
  }
  function Wn(i) {
    m(it, i, !0), m(_, i === "align" ? "对齐模式：默认只显示当前层" : "交叠预览：当前层突出、下一业务层淡显", !0), pe();
  }
  function za(i, o) {
    i.stopPropagation();
    const f = new Set(a(Yt)), v = new Set(a(Ge));
    cr(o) ? (f.add(o), v.delete(o)) : (f.delete(o), v.add(o)), m(Yt, f, !0), m(Ge, v, !0), m(_, "显示设置仅保存在浏览器，不改变保存数据"), pe();
  }
  function Ya(i, o) {
    i.stopPropagation(), m(De, a(De) === o ? "" : o, !0), m(_, a(De) ? "已单独显示一层" : "已退出单层显示", !0), pe();
  }
  function qa() {
    m(W, 1), m(je, 0), m(Ve, 0), m(_, "画布视图已复位（不影响保存数据）"), pe();
  }
  function zn(i) {
    const o = Te(a(W) * i, 0.2, 8), f = fe() / 2, v = we() / 2, g = (f - a(je)) / a(W), x = (v - a(Ve)) / a(W);
    m(je, f - g * o), m(Ve, v - x * o), m(W, o, !0), m(_, "画布视图 " + Math.round(a(W) * 100) + "%（仅本地）"), pe();
  }
  function Za(i, o) {
    return i === "x" ? Object.assign(Object.assign({}, a(A)), { center_x: o }) : i === "y" ? Object.assign(Object.assign({}, a(A)), { center_y: o }) : i === "scale" ? Object.assign(Object.assign({}, a(A)), { scale: Te(o / 100, 0.01, 20) }) : i === "rotation" ? Object.assign(Object.assign({}, a(A)), { rotation_deg: fr(o) }) : Object.assign(Object.assign({}, a(A)), { preview_alpha: Te(o / 100, 0, 1) });
  }
  function Wr(i) {
    const o = Number(a(at)[i]);
    if (m(lr, ""), !Number.isFinite(o)) {
      Pt();
      return;
    }
    const f = Se(a(A)), v = Se(Za(i, o)), g = Math.abs(v.center_x - f.center_x) < 1e-9 && Math.abs(v.center_y - f.center_y) < 1e-9 && Math.abs(v.scale - f.scale) < 1e-12 && Math.abs(v.rotation_deg - f.rotation_deg) < 1e-9 && Math.abs(v.preview_alpha - f.preview_alpha) < 1e-12;
    if (ze(v), g) {
      pe();
      return;
    }
    vr("numeric", de() + " 参数已更新");
  }
  function Ja(i, o) {
    m(at, Object.assign(Object.assign({}, a(at)), { [o]: i.currentTarget.value }), !0);
  }
  function $a(i, o) {
    const f = o.key === "ArrowUp" ? 1 : -1, v = i === "x" || i === "y" || i === "alpha" ? 1 : 0.1, g = i === "x" || i === "y" || i === "alpha" ? 0.1 : 0.01, x = i === "x" || i === "y" ? 10 : i === "alpha" ? 5 : 1;
    return f * (o.shiftKey ? g : o.altKey ? x : v);
  }
  function Ka(i, o) {
    if (i.key === "Escape") {
      i.preventDefault(), m(lr, ""), Pt(), i.currentTarget.blur();
      return;
    }
    if (i.key === "Enter") {
      i.preventDefault(), Wr(o), i.currentTarget.blur();
      return;
    }
    if (i.key === "ArrowUp" || i.key === "ArrowDown") {
      i.preventDefault();
      const f = ne(a(at)[o], 0);
      m(at, Object.assign(Object.assign({}, a(at)), { [o]: String(f + $a(o, i)) }), !0), Wr(o);
    }
  }
  function Qa(i) {
    i.code === "Space" && !(i.target instanceof HTMLInputElement) && !(i.target instanceof HTMLTextAreaElement) && m(ur, !0);
  }
  function es(i) {
    i.code === "Space" && m(ur, !1);
  }
  function Yn() {
    m(Ee, ""), m(Re, ""), m(He, "before");
  }
  function ts(i, o) {
    Ot() && (m(Ee, o, !0), m(Re, ""), m(He, "before"), m(nt, !1), i.dataTransfer && (i.dataTransfer.effectAllowed = "move", i.dataTransfer.setData("text/plain", o)));
  }
  function rs(i, o) {
    if (!Ot() || !a(Ee)) return;
    i.preventDefault(), i.dataTransfer && (i.dataTransfer.dropEffect = "move");
    const v = i.currentTarget.getBoundingClientRect();
    m(Re, o, !0), m(He, i.clientY < v.top + v.height / 2 ? "before" : "after", !0);
  }
  function ns(i, o) {
    var f;
    if (!Ot()) return;
    i.preventDefault();
    const v = a(Ee) || ((f = i.dataTransfer) === null || f === void 0 ? void 0 : f.getData("text/plain")) || "", g = a(Re) === o ? a(He) : "before", x = as(v, o, g);
    Yn(), x && (m(nt, !0), pr("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", !1, !0), setTimeout(
      () => {
        m(nt, !1);
      },
      0
    ));
  }
  function is() {
    Yn();
  }
  function as(i, o, f) {
    if (!Ot() || i === o) return !1;
    const v = a(b).group_view, g = Xe(), x = g.findIndex((q) => k(q.group_id) === i), w = g.findIndex((q) => k(q.group_id) === o);
    if (!v || x < 0 || w < 0) return !1;
    const N = g.slice(), [C] = N.splice(x, 1);
    if (!C) return !1;
    const L = N.findIndex((q) => k(q.group_id) === o);
    if (L < 0) return !1;
    const R = L + (f === "after" ? 1 : 0);
    return N.splice(R, 0, C), N.some((q, j) => k(q.group_id) !== k(g[j].group_id)) ? (m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        group_view: Object.assign(Object.assign({}, v), { groups: N })
      }),
      !0
    ), !0) : !1;
  }
  function qn(i, o) {
    if (a(nt)) {
      m(nt, !1);
      return;
    }
    const f = i.target;
    f?.closest(".group-drag-handle") || Qt(o) && pr("select_group", "已选择 Label：" + de(), !1, !0);
  }
  function ss(i) {
    const o = i.currentTarget.value;
    Qt(o) && pr("select_group", "已选择 Label：" + de(), !1, !0);
  }
  function Zn(i) {
    m(Mt, !0), m(G, "grabbing"), qt = {
      client_x: i.clientX,
      client_y: i.clientY,
      pan_x: a(je),
      pan_y: a(Ve)
    }, h.setPointerCapture(i.pointerId);
  }
  function os(i) {
    if (i.button !== 0) return;
    It();
    const o = Kt(i);
    if (a(ur)) {
      Zn(i);
      return;
    }
    if (me() && jn(o.x, o.y)) {
      m(G, "grabbing"), m(ee, !0), ae = {
        angle: Math.atan2(o.y - a(A).center_y, o.x - a(A).center_x) * 180 / Math.PI,
        rotation: Number(a(A).rotation_deg || 0)
      }, h.setPointerCapture(i.pointerId);
      return;
    }
    let f = !1;
    if (me() && z()) {
      const v = Xr(o.x, o.y);
      v && (Qt(v), f = !0);
    } else me() && (f = Dn(o.x, o.y));
    if (!f) {
      Zn(i), m(_, "正在平移画布视图（仅本地）");
      return;
    }
    m(G, "grabbing"), m(F, !0), tt = {
      x: o.x,
      y: o.y,
      center_x: Number(a(A).center_x || 0),
      center_y: Number(a(A).center_y || 0)
    }, h.setPointerCapture(i.pointerId);
  }
  function zr(i) {
    if (a(ur)) {
      m(G, "grab");
      return;
    }
    if (!me()) {
      m(G, "grab");
      return;
    }
    const o = z() ? !!Xr(i.x, i.y) : Dn(i.x, i.y);
    m(G, (jn(i.x, i.y) || o, "grab"), !0);
  }
  function us(i) {
    if (a(Mt)) {
      const f = h.getBoundingClientRect();
      m(je, qt.pan_x + (i.clientX - qt.client_x) * fe() / Math.max(1, f.width)), m(Ve, qt.pan_y + (i.clientY - qt.client_y) * we() / Math.max(1, f.height)), m(_, "正在平移画布视图（仅本地）"), pe();
      return;
    }
    const o = Kt(i);
    if (!a(F) && !a(ee)) {
      zr(o);
      return;
    }
    if (m(G, "grabbing"), a(F))
      ze(Object.assign(Object.assign({}, a(A)), {
        center_x: tt.center_x + o.x - tt.x,
        center_y: tt.center_y + o.y - tt.y
      })), m(_, "正在拖动 " + (z() ? de() : "版图") + "（本地）");
    else if (a(ee)) {
      const f = Math.atan2(o.y - a(A).center_y, o.x - a(A).center_x) * 180 / Math.PI;
      ze(Object.assign(Object.assign({}, a(A)), {
        rotation_deg: fr(ae.rotation + f - ae.angle)
      })), m(_, "正在旋转 " + (z() ? de() : "版图") + "（本地）");
    }
    pe();
  }
  function ls() {
    !a(F) && !a(ee) && !a(Mt) && m(G, "grab");
  }
  function Jn(i) {
    if (a(Mt)) {
      m(Mt, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      m(_, "画布视图已调整（仅本地，不影响保存数据）"), zr(Kt(i));
      return;
    }
    if (a(F) || a(ee)) {
      m(F, !1), m(ee, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      zr(Kt(i));
      const f = (z() ? de() : "Canvas") + " 已在本地更新，停止操作 250ms 后静默同步";
      z() ? vr("canvas", f) : bt("canvas", f);
    }
  }
  function $n(i) {
    return i.deltaMode === WheelEvent.DOM_DELTA_LINE ? i.deltaY * 16 : i.deltaMode === WheelEvent.DOM_DELTA_PAGE ? i.deltaY * Math.max(1, h.getBoundingClientRect().height) : i.deltaY;
  }
  function fs(i) {
    i.preventDefault();
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * fe(), v = (i.clientY - o.top) / Math.max(1, o.height) * we();
    if (i.ctrlKey || i.metaKey) {
      const J = (f - a(je)) / a(W), q = (v - a(Ve)) / a(W), j = Te(a(W) * Math.exp(-$n(i) * 15e-4), 0.2, 8);
      m(je, f - J * j), m(Ve, v - q * j), m(W, j, !0), m(_, "画布视图 " + Math.round(a(W) * 100) + "%（仅本地）"), pe();
      return;
    }
    if (!me()) return;
    const g = Kt(i);
    if (z()) {
      const J = Xr(g.x, g.y);
      J && Qt(J);
    }
    const x = jr(g.x, g.y, a(A)), w = i.shiftKey ? 0.1 : i.altKey ? 5 : 1, N = Math.exp(-$n(i) * s * w), C = Te(Number(a(A).scale || 1) * N, 0.01, 20);
    let L = Object.assign(Object.assign({}, a(A)), { scale: C });
    const R = $t(x.x, x.y, L);
    L = Object.assign(Object.assign({}, L), {
      center_x: Number(L.center_x || 0) + g.x - R.x,
      center_y: Number(L.center_y || 0) + g.y - R.y
    }), ze(L), vr("canvas", "本地缩放 " + (C * 100).toFixed(2) + "%；停止 250ms 后同步"), pe();
  }
  function cs() {
    me() && (ze(Object.assign(Object.assign({}, a(A)), {
      center_x: fe() / 2,
      center_y: we() / 2,
      scale: 1,
      rotation_deg: 0
    })), bt("reset", "居中归一 " + (z() ? de() : "版图") + "：scale=1，rotation=0"));
  }
  function hs() {
    me() && (ze(Object.assign(Object.assign({}, a(A)), { center_x: fe() / 2, center_y: we() / 2 })), bt("center", "居中 " + (z() ? de() : "版图") + "：保留缩放和旋转"));
  }
  function ds() {
    if (!me()) return;
    const [i, o, f, v] = Bn(), g = Math.max(1, f - i + 1), x = Math.max(1, v - o + 1), w = Te(Math.min(fe() / g, we() / x) * 0.9, 0.01, 20);
    ze(Object.assign(Object.assign({}, a(A)), {
      center_x: fe() / 2,
      center_y: we() / 2,
      scale: w
    })), bt("fit", "适配 " + (z() ? de() : "版图") + "：scale=" + w.toFixed(3));
  }
  function ps() {
    if (!me()) return;
    const i = hr(), o = Math.min(...i.map((C) => C.x)), f = Math.max(...i.map((C) => C.x)), v = Math.min(...i.map((C) => C.y)), g = Math.max(...i.map((C) => C.y));
    let x = 0, w = 0;
    const N = Math.max(20, fe() * 0.03);
    if (f < N ? x = N - f : o > fe() - N && (x = fe() - N - o), g < N ? w = N - g : v > we() - N && (w = we() - N - v), x === 0 && w === 0) {
      m(_, (z() ? de() : "版图") + " 已经在视野内");
      return;
    }
    ze(Object.assign(Object.assign({}, a(A)), {
      center_x: Number(a(A).center_x || 0) + x,
      center_y: Number(a(A).center_y || 0) + w
    })), bt("bring_into_view", "已找回 " + (z() ? de() : "版图") + " 到视野内");
  }
  V("keydown", di, Qa), V("keyup", di, es);
  {
    let i = xt(() => a(F) || a(ee) || a(Mt) ? "focus" : "base");
    Fl(e, {
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
        var v = Ql(), g = $(v);
        {
          var x = (se) => {
            var ve = zl(), oe = Y($(ve), 2);
            xr(oe, 21, Xe, yr, (ft, ct) => {
              var Ue = jl(), yt = $(Ue), ie = {};
              Ye(
                (er, Pe) => {
                  _e(yt, er), ie !== (ie = Pe) && (Ue.value = (Ue.__value = Pe) ?? "");
                },
                [
                  () => a(ct).label || "Label " + String(a(ct).group_id),
                  () => k(a(ct).group_id)
                ]
              ), le(ft, Ue);
            });
            var ue;
            ta(oe);
            var Ae = Y(oe, 2), Ne = $(Ae);
            let ut;
            var Me = Y(Ne, 2);
            let lt;
            var mt = Y(Ae, 2);
            {
              var et = (ft) => {
                var ct = Wl(), Ue = Y($(ct), 2);
                xr(Ue, 21, Xe, yr, (yt, ie, er) => {
                  var Pe = Xl();
                  let ri;
                  var br = $(Pe);
                  Ut(br, "draggable", !0);
                  var ni = Y(br, 2);
                  ni.textContent = er + 1;
                  var ii = Y(ni, 2), ai = $(ii), Ts = Y(ai);
                  {
                    var ws = (re) => {
                      var qr = Vl();
                      le(re, qr);
                    }, As = xt(() => k(a(ie).group_id) === a(T));
                    qe(Ts, (re) => {
                      a(As) && re(ws);
                    });
                  }
                  var mr = Y(ii, 2), Ms = $(mr), si = Y(mr, 2);
                  let oi;
                  Ye(
                    (re, qr, Os, Hs, Is) => {
                      ri = Et(Pe, 1, "group-reorder-item svelte-r41nsf", null, ri, re), _e(ai, `${qr ?? ""} `), Ut(mr, "title", Os), _e(Ms, Hs), oi = Et(si, 1, "solo-button svelte-r41nsf", null, oi, Is);
                    },
                    [
                      () => ({
                        active: k(a(ie).group_id) === a(T),
                        "drop-before": a(Re) === k(a(ie).group_id) && a(He) === "before",
                        "drop-after": a(Re) === k(a(ie).group_id) && a(He) === "after"
                      }),
                      () => a(ie).label || "Label " + String(a(ie).group_id),
                      () => cr(k(a(ie).group_id)) ? "隐藏此层" : "显示此层",
                      () => cr(k(a(ie).group_id)) ? "👁" : "—",
                      () => ({
                        active: a(De) === k(a(ie).group_id)
                      })
                    ]
                  ), V("dragstart", br, (re) => ts(re, k(a(ie).group_id))), V("dragend", br, is), V("click", mr, (re) => za(re, k(a(ie).group_id))), V("click", si, (re) => Ya(re, k(a(ie).group_id))), V("click", Pe, (re) => qn(re, k(a(ie).group_id))), V("keydown", Pe, (re) => {
                    (re.key === "Enter" || re.key === " ") && (re.preventDefault(), qn(re, k(a(ie).group_id)));
                  }), V("dragover", Pe, (re) => rs(re, k(a(ie).group_id))), V("drop", Pe, (re) => ns(re, k(a(ie).group_id))), le(yt, Pe);
                }), le(ft, ct);
              }, _t = xt(() => Ot());
              qe(mt, (ft) => {
                a(_t) && ft(et);
              });
            }
            Ye(() => {
              ue !== (ue = a(T)) && (oe.value = (oe.__value = a(T)) ?? "", Hr(oe, a(T))), ut = Et(Ne, 1, "svelte-r41nsf", null, ut, { active: a(it) === "align" }), lt = Et(Me, 1, "svelte-r41nsf", null, lt, { active: a(it) === "overlap" });
            }), V("change", oe, ss), V("click", Ne, () => Wn("align")), V("click", Me, () => Wn("overlap")), le(se, ve);
          }, w = xt(() => z() && !Ht());
          qe(g, (se) => {
            a(w) && se(x);
          });
        }
        var N = Y(g, 2), C = $(N), L = Y(C, 2), R = Y(L, 2), J = Y(R, 2), q = Y(J, 2), j = Y(q, 2), Ie = $(j), Ke = Y(j, 2), Qe = Y(N, 2);
        {
          var vs = (se) => {
            var ve = ql();
            xr(
              ve,
              20,
              () => [
                ["x", "X", "px"],
                ["y", "Y", "px"],
                ["scale", "缩放", "%"],
                ["rotation", "旋转", "°"],
                ["alpha", "透明度", "%"]
              ],
              yr,
              (oe, ue) => {
                var Ae = Yl(), Ne = $(Ae), ut = $(Ne), Me = Y(Ne, 2), lt = Y(Me, 2), mt = $(lt);
                Ye(() => {
                  _e(ut, ue[1]), Jo(Me, a(at)[ue[0]]), _e(mt, ue[2]);
                }), V("focus", Me, () => m(lr, ue[0], !0)), V("input", Me, (et) => Ja(et, ue[0])), V("keydown", Me, (et) => Ka(et, ue[0])), V("blur", Me, () => Wr(ue[0])), le(oe, Ae);
              }
            ), Ye((oe) => Ut(ve, "aria-label", oe), [() => Ht() ? "当前标注精细参数" : "当前层精细参数"]), le(se, ve);
          }, gs = xt(() => me());
          qe(Qe, (se) => {
            a(gs) && se(vs);
          });
        }
        var Kn = Y(Qe, 2);
        {
          var bs = (se) => {
            var ve = $l();
            let oe;
            var ue = $(ve), Ae = $(ue), Ne = $(Ae), ut = Y(Ae, 2), Me = $(ut), lt = Y(ue, 2);
            {
              var mt = (_t) => {
                var ft = Jl();
                xr(ft, 21, Ln, yr, (ct, Ue) => {
                  var yt = Zl(), ie = $(yt);
                  Ye(
                    (er, Pe) => _e(ie, `应用 ${a(Ue).candidate_id ?? ""}
								· 支持率 ${er ?? ""}%
								· 分数 ${Pe ?? ""}`),
                    [
                      () => Math.round(ne(a(Ue).metrics?.central_support, 0) * 100),
                      () => ne(a(Ue).metrics?.score, 0).toFixed(3)
                    ]
                  ), V("click", yt, () => Rn(a(Ue))), le(ct, yt);
                }), le(_t, ft);
              }, et = xt(() => Ln().length);
              qe(lt, (_t) => {
                a(et) && _t(mt);
              });
            }
            Ye(() => {
              oe = Et(ve, 1, "alignment-proposal svelte-r41nsf", null, oe, { ambiguous: a(he).ambiguous === !0 }), _e(Ne, a(he).ambiguous ? "存在周期歧义" : "粗对齐建议"), _e(Me, a(he).status);
            }), le(se, ve);
          };
          qe(Kn, (se) => {
            a(he) && se(bs);
          });
        }
        var Qn = Y(Kn, 2), ot = $(Qn);
        aa(ot, (se) => h = se, () => h);
        var ei = Y(Qn, 2);
        {
          var ms = (se) => {
            var ve = Kl(), oe = $(ve), ue = $(oe), Ae = Y(oe, 2), Ne = $(Ae), ut = Y(Ae, 2), Me = $(ut);
            Ye(
              (lt, mt, et, _t) => {
                _e(ue, `当前：${lt ?? ""}`), _e(Ne, `下一层：${mt ?? ""}`), _e(Me, `${et ?? ""} × ${_t ?? ""}`);
              },
              [
                () => de(),
                () => I.get(Jt())?.view.label,
                () => de(),
                () => I.get(Jt())?.view.label
              ]
            ), le(se, ve);
          }, _s = xt(() => z() && !Ht() && a(it) === "overlap" && Jt());
          qe(ei, (se) => {
            a(_s) && se(ms);
          });
        }
        var ys = Y(ei, 2), Yr = $(ys);
        let ti;
        var xs = $(Yr), Es = Y(Yr, 2), Ss = $(Es);
        Ye(
          (se, ve, oe, ue, Ae, Ne) => {
            Or(v, se), C.disabled = ve, L.disabled = oe, R.disabled = ue, J.disabled = Ae, _e(Ie, `${Ne ?? ""}%`), Or(ot, "cursor:" + a(G)), ti = Et(Yr, 1, "svelte-r41nsf", null, ti, { dirty: Le > te }), _e(xs, Le > te ? be === null ? "本地修改" : "同步中" : "已同步"), _e(Ss, a(_));
          },
          [
            () => "min-height:" + Oa(d.props.height),
            () => !me(),
            () => !me(),
            () => !me(),
            () => !me(),
            () => Math.round(a(W) * 100)
          ]
        ), V("click", C, cs), V("click", L, hs), V("click", R, ds), V("click", J, ps), V("click", q, () => zn(0.8)), V("click", j, qa), V("click", Ke, () => zn(1.25)), V("pointerdown", ot, os), V("pointermove", ot, us), V("pointerup", ot, Jn), V("pointercancel", ot, Jn), V("pointerleave", ot, ls), V("wheel", ot, fs), le(o, v);
      },
      $$slots: { default: !0 }
    });
  }
  Zi();
}
export {
  tf as default
};
