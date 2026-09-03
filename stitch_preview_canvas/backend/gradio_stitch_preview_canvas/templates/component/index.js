import { i as un, g as Kn, o as Qi, n as lt, u as ue, s as Ki, r as Xr, m as yt, a as b, b as o, t as fn, d as $i, q as ea, c as $n, e as Et, f as br, h as dr, j as ta, T as ra, k as na, l as pr, p as xt, v as cn, w as wt, x as ei, y as ti, z as ri, A as $t, E as _r, B as Nt, C as ni, D as Le, F as _n, G as ia, H as ii, I as hn, J as aa, K as yn, L as sa, M as oa, N as Qe, O as ai, P as Br, Q as la, R as ua, S as fa, U as ca, V as si, W as dn, X as xn, Y as En, Z as ha, _ as da, $ as pa, a0 as ma, a1 as ga, a2 as va, a3 as ba, a4 as _a, a5 as pn, a6 as ya, a7 as st, a8 as er, a9 as xa, aa as Ea, ab as wa, ac as Ta, ad as Sa, ae as Aa, af as mn, ag as Ha, ah as Pa, ai as Be, aj as qr, ak as Wr, al as Ia, am as Ma, an as Qt, ao as Oa, ap as Ba, aq as La, ar as Na, as as Ca, at as oi, au as qt, av as z, aw as Ra, ax as wn, ay as Da, az as xe, aA as yr, aB as xr, aC as Z, aD as Ot, aE as te, aF as ka, aG as ae, aH as ye, aI as ke, aJ as Ua } from "./render-DoYhCszp.js";
function li(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const Fa = [];
function Ga(e, t = !1, r = !1) {
  return fr(e, /* @__PURE__ */ new Map(), "", Fa, null, r);
}
function fr(e, t, r, n, i = null, a = !1) {
  if (typeof e == "object" && e !== null) {
    var u = t.get(e);
    if (u !== void 0) return u;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (un(e)) {
      var l = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, l), i !== null && t.set(i, l);
      for (var c = 0; c < e.length; c += 1) {
        var f = e[c];
        c in e && (l[c] = fr(f, t, r, n, null, a));
      }
      return l;
    }
    if (Kn(e) === Qi) {
      l = {}, t.set(e, l), i !== null && t.set(i, l);
      for (var p of Object.keys(e))
        l[p] = fr(
          // @ts-expect-error
          e[p],
          t,
          r,
          n,
          null,
          a
        );
      return l;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !a)
      return fr(
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
function gn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), lt;
  const n = ue(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Mt = [];
function ja(e, t) {
  return {
    subscribe: tr(e, t).subscribe
  };
}
function tr(e, t = lt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(l) {
    if (Ki(e, l) && (e = l, r)) {
      const c = !Mt.length;
      for (const f of n)
        f[1](), Mt.push(f, e);
      if (c) {
        for (let f = 0; f < Mt.length; f += 2)
          Mt[f][0](Mt[f + 1]);
        Mt.length = 0;
      }
    }
  }
  function a(l) {
    i(l(
      /** @type {T} */
      e
    ));
  }
  function u(l, c = lt) {
    const f = [l, c];
    return n.add(f), n.size === 1 && (r = t(i, a) || lt), l(
      /** @type {T} */
      e
    ), () => {
      n.delete(f), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: u };
}
function Ft(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return ja(r, (u, l) => {
    let c = !1;
    const f = [];
    let p = 0, _ = lt;
    const x = () => {
      if (p)
        return;
      _();
      const v = t(n ? f[0] : f, u, l);
      a ? u(v) : _ = typeof v == "function" ? v : lt;
    }, S = i.map(
      (v, A) => gn(
        v,
        (P) => {
          f[A] = P, p &= ~(1 << A), c && x();
        },
        () => {
          p |= 1 << A;
        }
      )
    );
    return c = !0, x(), function() {
      Xr(S), _(), c = !1;
    };
  });
}
function Va(e) {
  let t;
  return gn(e, (r) => t = r)(), t;
}
let or = !1, Yr = /* @__PURE__ */ Symbol("unmounted");
function Tn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: yt(void 0),
    unsubscribe: lt
  };
  if (n.store !== e && !(Yr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = lt;
    else {
      var i = !0;
      n.unsubscribe = gn(e, (a) => {
        i ? n.source.v = a : b(n.source, a);
      }), i = !1;
    }
  return e && Yr in r ? Va(e) : o(n.source);
}
function za() {
  const e = {};
  function t() {
    fn(() => {
      for (var r in e)
        e[r].unsubscribe();
      $i(e, Yr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Xa(e) {
  var t = or;
  try {
    return or = !1, [e(), or];
  } finally {
    or = t;
  }
}
function qa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, ea(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Wa = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Ya(e) {
  return (
    /** @type {string} */
    Wa?.createHTML(e) ?? e
  );
}
function ui(e) {
  var t = $n("template");
  return t.innerHTML = Ya(e.replaceAll("<!>", "<!---->")), t.content;
}
function Rt(e, t) {
  var r = (
    /** @type {Effect} */
    br
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function de(e, t) {
  var r = (t & ra) !== 0, n = (t & na) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = ui(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    dr(i)));
    var u = (
      /** @type {TemplateNode} */
      n || ta ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var l = (
        /** @type {TemplateNode} */
        dr(u)
      ), c = (
        /** @type {TemplateNode} */
        u.lastChild
      );
      Rt(l, c);
    } else
      Rt(u, u);
    return u;
  };
}
// @__NO_SIDE_EFFECTS__
function Za(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var u = (
        /** @type {DocumentFragment} */
        ui(i)
      ), l = (
        /** @type {Element} */
        dr(u)
      );
      a = /** @type {Element} */
      dr(l);
    }
    var c = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Rt(c, c), c;
  };
}
// @__NO_SIDE_EFFECTS__
function fi(e, t) {
  return /* @__PURE__ */ Za(e, t, "svg");
}
function qe(e = "") {
  {
    var t = Et(e + "");
    return Rt(t, t), t;
  }
}
function Lt() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = Et();
  return e.append(t, r), Rt(t, r), e;
}
function U(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Er {
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
        pr(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (pr(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, u] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const l = this.#e.get(u);
        l && (xt(l.effect), this.#e.delete(u));
      }
      for (const [a, u] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const l = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var f = document.createDocumentFragment();
            ti(u, f), f.append(Et()), this.#e.set(a, { effect: u, fragment: f });
          } else
            xt(u);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), cn(u, l, !1)) : l();
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
      ei
    ), i = ri();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), u = Et();
        a.append(u), this.#e.set(t, {
          effect: wt(() => r(u)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          wt(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [l, c] of this.#r)
        l === t ? n.unskip_effect(c) : n.skip_effect(c);
      for (const [l, c] of this.#e)
        l === t ? n.unskip_effect(c.effect) : n.skip_effect(c.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Ja(e, t, ...r) {
  var n = new Er(e);
  $t(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, _r);
}
function ci(e) {
  Nt === null && li(), ni && Nt.l !== null ? Ka(Nt).m.push(e) : Le(() => {
    const t = ue(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Qa(e) {
  Nt === null && li(), ci(() => () => ue(e));
}
function Ka(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function $(e, t, r = !1) {
  var n = new Er(e), i = r ? _r : 0;
  function a(u, l) {
    n.ensure(u, l);
  }
  $t(() => {
    var u = !1;
    t((l, c = 0) => {
      u = !0, a(c, l);
    }), u || a(-1, null);
  }, i);
}
function Sn(e, t) {
  return t;
}
function $a(e, t, r) {
  for (var n = [], i = t.length, a, u = t.length, l = 0; l < i; l++) {
    let _ = t[l];
    cn(
      _,
      () => {
        if (a) {
          if (a.pending.delete(_), a.done.add(_), a.pending.size === 0) {
            var x = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Zr(e, hn(a.done)), x.delete(a), x.size === 0 && (e.outrogroups = null);
          }
        } else
          u -= 1;
      },
      !1
    );
  }
  if (u === 0) {
    var c = n.length === 0 && r !== null && e.pending.size === 0;
    if (c) {
      var f = (
        /** @type {Element} */
        r
      ), p = (
        /** @type {Element} */
        f.parentNode
      );
      ua(p), p.append(f), e.items.clear();
    }
    Zr(e, t, !c);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Zr(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const u of e.pending.values())
      for (const l of u)
        n.add(
          /** @type {EachItem} */
          e.items.get(l).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var a = t[i];
    if (n?.has(a)) {
      a.f |= Qe;
      const u = document.createDocumentFragment();
      ti(a, u);
    } else
      xt(t[i], r);
  }
}
var An;
function Hn(e, t, r, n, i, a = null) {
  var u = e, l = /* @__PURE__ */ new Map(), c = null, f = ii(() => {
    var d = r();
    return (
      /** @type {V[]} */
      un(d) ? d : d == null ? [] : hn(d)
    );
  }), p, _ = /* @__PURE__ */ new Map(), x = !0;
  function S(d) {
    (P.effect.f & ai) === 0 && (P.pending.delete(d), P.fallback = c, es(P, p, u, t, n), c !== null && (p.length === 0 ? (c.f & Qe) === 0 ? pr(c) : (c.f ^= Qe, Zt(c, null, u)) : cn(c, () => {
      c = null;
    })));
  }
  function v(d) {
    P.pending.delete(d);
  }
  var A = $t(() => {
    p = /** @type {V[]} */
    o(f);
    for (var d = p.length, y = /* @__PURE__ */ new Set(), H = (
      /** @type {Batch} */
      ei
    ), E = ri(), w = 0; w < d; w += 1) {
      var B = p[w], M = n(B, w), L = x ? null : l.get(M);
      L ? (L.v && _n(L.v, B), L.i && _n(L.i, w), E && H.unskip_effect(L.e)) : (L = ts(
        l,
        x ? u : An ??= Et(),
        B,
        M,
        w,
        i,
        t,
        r
      ), x || (L.e.f |= Qe), l.set(M, L)), y.add(M);
    }
    if (d === 0 && a && !c && (x ? c = wt(() => a(u)) : (c = wt(() => a(An ??= Et())), c.f |= Qe)), d > y.size && ia(), !x)
      if (_.set(H, y), E) {
        for (const [R, C] of l)
          y.has(R) || H.skip_effect(C.e);
        H.oncommit(S), H.ondiscard(v);
      } else
        S(H);
    o(f);
  }), P = { effect: A, items: l, pending: _, outrogroups: null, fallback: c };
  x = !1;
}
function Wt(e) {
  for (; e !== null && (e.f & la) === 0; )
    e = e.next;
  return e;
}
function es(e, t, r, n, i) {
  var a = t.length, u = e.items, l = Wt(e.effect.first), c, f = null, p = [], _ = [], x, S, v, A;
  for (A = 0; A < a; A += 1) {
    if (x = t[A], S = i(x, A), v = /** @type {EachItem} */
    u.get(S).e, e.outrogroups !== null)
      for (const L of e.outrogroups)
        L.pending.delete(v), L.done.delete(v);
    if ((v.f & Br) !== 0 && pr(v), (v.f & Qe) !== 0)
      if (v.f ^= Qe, v === l)
        Zt(v, null, r);
      else {
        var P = f ? f.next : l;
        v === e.effect.last && (e.effect.last = v.prev), v.prev && (v.prev.next = v.next), v.next && (v.next.prev = v.prev), at(e, f, v), at(e, v, P), Zt(v, P, r), f = v, p = [], _ = [], l = Wt(f.next);
        continue;
      }
    if (v !== l) {
      if (c !== void 0 && c.has(v)) {
        if (p.length < _.length) {
          var d = _[0], y;
          f = d.prev;
          var H = p[0], E = p[p.length - 1];
          for (y = 0; y < p.length; y += 1)
            Zt(p[y], d, r);
          for (y = 0; y < _.length; y += 1)
            c.delete(_[y]);
          at(e, H.prev, E.next), at(e, f, H), at(e, E, d), l = d, f = E, A -= 1, p = [], _ = [];
        } else
          c.delete(v), Zt(v, l, r), at(e, v.prev, v.next), at(e, v, f === null ? e.effect.first : f.next), at(e, f, v), f = v;
        continue;
      }
      for (p = [], _ = []; l !== null && l !== v; )
        (c ??= /* @__PURE__ */ new Set()).add(l), _.push(l), l = Wt(l.next);
      if (l === null)
        continue;
    }
    (v.f & Qe) === 0 && p.push(v), f = v, l = Wt(v.next);
  }
  if (e.outrogroups !== null) {
    for (const L of e.outrogroups)
      L.pending.size === 0 && (Zr(e, hn(L.done)), e.outrogroups?.delete(L));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (l !== null || c !== void 0) {
    var w = [];
    if (c !== void 0)
      for (v of c)
        (v.f & Br) === 0 && w.push(v);
    for (; l !== null; )
      (l.f & Br) === 0 && l !== e.fallback && w.push(l), l = Wt(l.next);
    var B = w.length;
    if (B > 0) {
      var M = null;
      $a(e, w, M);
    }
  }
}
function ts(e, t, r, n, i, a, u, l) {
  var c = (u & sa) !== 0 ? (u & oa) === 0 ? yt(r, !1, !1) : yn(r) : null, f = (u & aa) !== 0 ? yn(i) : null;
  return {
    v: c,
    i: f,
    e: wt(() => (a(t, c ?? r, f ?? i, l), () => {
      e.delete(n);
    }))
  };
}
function Zt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Qe) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var u = (
        /** @type {TemplateNode} */
        fa(n)
      );
      if (a.before(n), n === i)
        return;
      n = u;
    }
}
function at(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Jr(e, t, r, n, i) {
  var a = t.$$slots?.[r], u = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], u = !0), a === void 0 || a(e, u ? () => n : n);
}
function rs(e, t, r) {
  var n = new Er(e);
  $t(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, _r);
}
const ns = () => performance.now(), Ue = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => ns(),
  tasks: /* @__PURE__ */ new Set()
};
function hi() {
  const e = Ue.now();
  Ue.tasks.forEach((t) => {
    t.c(e) || (Ue.tasks.delete(t), t.f());
  }), Ue.tasks.size !== 0 && Ue.tick(hi);
}
function is(e) {
  let t;
  return Ue.tasks.size === 0 && Ue.tick(hi), {
    promise: new Promise((r) => {
      Ue.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Ue.tasks.delete(t);
    }
  };
}
function as(e, t, r, n, i, a) {
  var u = null, l = (
    /** @type {TemplateNode} */
    e
  ), c = new Er(l, !1);
  $t(() => {
    const f = t() || null;
    var p = f === "svg" ? ca : void 0;
    if (f === null) {
      c.ensure(null, null);
      return;
    }
    return c.ensure(f, (_) => {
      if (f) {
        if (u = $n(f, p), Rt(u, u), n) {
          var x = null, S = u.appendChild(Et());
          n(u, S), x?.remove();
        }
        br.nodes.end = u, _.before(u);
      }
    }), () => {
    };
  }, _r), fn(() => {
  });
}
function ss(e, t) {
  var r = void 0, n;
  si(() => {
    r !== (r = t()) && (n && (xt(n), n = null), r && (n = wt(() => {
      dn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function di(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = di(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function os() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = di(e)) && (n && (n += " "), n += t);
  return n;
}
function ls(e) {
  return typeof e == "object" ? os(e) : e ?? "";
}
const Pn = [...`\x20\t
\r\f \v\uFEFF`];
function us(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, u = 0; (u = n.indexOf(i, u)) >= 0; ) {
          var l = u + a;
          (u === 0 || Pn.includes(n[u - 1])) && (l === n.length || Pn.includes(n[l])) ? n = (u === 0 ? "" : n.substring(0, u)) + n.substring(l + 1) : u = l;
        }
  }
  return n === "" ? null : n;
}
function In(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function Lr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function fs(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\/\*.*?\*\//g, "").trim();
      var a = !1, u = 0, l = !1, c = [];
      n && c.push(...Object.keys(n).map(Lr)), i && c.push(...Object.keys(i).map(Lr));
      var f = 0, p = -1;
      const A = e.length;
      for (var _ = 0; _ < A; _++) {
        var x = e[_];
        if (l ? x === "/" && e[_ - 1] === "*" && (l = !1) : a ? a === x && (a = !1) : x === "/" && e[_ + 1] === "*" ? l = !0 : x === '"' || x === "'" ? a = x : x === "(" ? u++ : x === ")" && u--, !l && a === !1 && u === 0) {
          if (x === ":" && p === -1)
            p = _;
          else if (x === ";" || _ === A - 1) {
            if (p !== -1) {
              var S = Lr(e.substring(f, p).trim());
              if (!c.includes(S)) {
                x !== ";" && _++;
                var v = e.substring(f, _).trim();
                r += " " + v + ";";
              }
            }
            f = _ + 1, p = -1;
          }
        }
      }
    }
    return n && (r += In(n)), i && (r += In(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function ut(e, t, r, n, i, a) {
  var u = (
    /** @type {any} */
    e[xn]
  );
  if (u !== r || u === void 0) {
    var l = us(r, n, a);
    l == null ? e.removeAttribute("class") : t ? e.className = l : e.setAttribute("class", l), e[xn] = r;
  } else if (a && i !== a)
    for (var c in a) {
      var f = !!a[c];
      (i == null || f !== !!i[c]) && e.classList.toggle(c, f);
    }
  return a;
}
function Nr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Fe(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[En]
  );
  if (i !== t) {
    var a = fs(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[En] = t;
  } else n && (Array.isArray(n) ? (Nr(e, r?.[0], n[0]), Nr(e, r?.[1], n[1], "important")) : Nr(e, r, n));
  return n;
}
function Qr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!un(t))
      return ha();
    for (var n of e.options)
      n.selected = t.includes(Mn(n));
    return;
  }
  for (n of e.options) {
    var i = Mn(n);
    if (da(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function cs(e) {
  var t = new MutationObserver(() => {
    "__value" in e && Qr(e, e.__value);
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
function Mn(e) {
  return "__value" in e ? e.__value : e.value;
}
const Jt = /* @__PURE__ */ Symbol("class"), Bt = /* @__PURE__ */ Symbol("style"), pi = /* @__PURE__ */ Symbol("is custom element"), mi = /* @__PURE__ */ Symbol("is html"), hs = pn ? "input" : "INPUT", ds = pn ? "option" : "OPTION", ps = pn ? "select" : "SELECT";
function ms(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function Ct(e, t, r, n) {
  var i = gi(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[pa] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && vi(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function gs(e, t, r, n, i = !1, a = !1) {
  var u = gi(e), l = u[pi], c = !u[mi], f = t || {}, p = e.nodeName === ds;
  for (var _ in t)
    !(_ in r) && _[0] + _[1] !== "$$" && (r[_] = null);
  r.class ? r.class = ls(r.class) : r.class = null, r[Bt] && (r.style ??= null);
  var x = vi(e);
  if (e.nodeName === hs && "type" in r && ("value" in r || "__value" in r)) {
    var S = r.type;
    (S !== f.type || S === void 0 && e.hasAttribute("type")) && (f.type = S, Ct(e, "type", S));
  }
  for (const E in r) {
    let w = r[E];
    if (p && E === "value" && w == null) {
      e.value = e.__value = "", f[E] = w;
      continue;
    }
    if (E === "class") {
      var v = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      ut(e, v, w, n, t?.[Jt], r[Jt]), f[E] = w, f[Jt] = r[Jt];
      continue;
    }
    if (E === "style") {
      Fe(e, w, t?.[Bt], r[Bt]), f[E] = w, f[Bt] = r[Bt];
      continue;
    }
    var A = f[E];
    if (!(w === A && !(w === void 0 && e.hasAttribute(E)))) {
      f[E] = w;
      var P = E[0] + E[1];
      if (P !== "$$")
        if (P === "on") {
          const B = {}, M = "$$" + E;
          let L = E.slice(2);
          var d = Ta(L);
          if (ya(L) && (L = L.slice(0, -7), B.capture = !0), !d && A) {
            if (w != null) continue;
            e.removeEventListener(L, f[M], B), f[M] = null;
          }
          if (d)
            st(L, e, w), er([L]);
          else if (w != null) {
            let R = function(C) {
              f[E].call(this, C);
            };
            f[M] = xa(L, e, R, B);
          }
        } else if (E === "style")
          Ct(e, E, w);
        else if (E === "autofocus")
          qa(
            /** @type {HTMLElement} */
            e,
            !!w
          );
        else if (!l && (E === "__value" || E === "value" && w != null))
          e.value = e.__value = w;
        else if (E === "selected" && p)
          ms(
            /** @type {HTMLOptionElement} */
            e,
            w
          );
        else {
          var y = E;
          c || (y = Ea(y));
          var H = y === "defaultValue" || y === "defaultChecked";
          if (w == null && !l && !H)
            if (u[E] = null, y === "value" || y === "checked") {
              let B = (
                /** @type {HTMLInputElement} */
                e
              );
              const M = t === void 0;
              if (y === "value") {
                let L = B.defaultValue;
                B.removeAttribute(y), B.defaultValue = L, B.value = B.__value = M ? L : null;
              } else {
                let L = B.defaultChecked;
                B.removeAttribute(y), B.defaultChecked = L, B.checked = M ? L : !1;
              }
            } else
              e.removeAttribute(E);
          else H || x.includes(y) && (l || typeof w != "string") ? (e[y] = w, y in u && (u[y] = wa)) : typeof w != "function" && Ct(e, y, w);
        }
    }
  }
  return f;
}
function vs(e, t, r = [], n = [], i = [], a, u = !1, l = !1) {
  ba(i, r, n, (c) => {
    var f = void 0, p = {}, _ = e.nodeName === ps, x = !1;
    if (si(() => {
      var v = t(...c.map(o)), A = gs(
        e,
        f,
        v,
        a,
        u,
        l
      );
      x && _ && "value" in v && Qr(
        /** @type {HTMLSelectElement} */
        e,
        v.value
      );
      for (let d of Object.getOwnPropertySymbols(p))
        v[d] || xt(p[d]);
      for (let d of Object.getOwnPropertySymbols(v)) {
        var P = v[d];
        d.description === _a && (!f || P !== f[d]) && (p[d] && xt(p[d]), p[d] = wt(() => ss(e, () => P))), A[d] = P;
      }
      f = A;
    }), _) {
      var S = (
        /** @type {HTMLSelectElement} */
        e
      );
      dn(() => {
        Qr(
          S,
          /** @type {Record<string | symbol, any>} */
          f.value,
          !0
        ), cs(S);
      });
    }
    x = !0;
  });
}
function gi(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[ma] ??= {
      [pi]: e.nodeName.includes("-"),
      [mi]: e.namespaceURI === ga
    }
  );
}
var On = /* @__PURE__ */ new Map();
function vi(e) {
  var t = e.getAttribute("is") || e.nodeName, r = On.get(t);
  if (r) return r;
  On.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = va(i);
    for (var u in n)
      n[u].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      u !== "innerHTML" && u !== "textContent" && u !== "innerText" && r.push(u);
    i = Kn(i);
  }
  return r;
}
function Cr(e, t) {
  return e === t || e?.[mn] === t;
}
function mr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Nt.r
  ), a = (
    /** @type {Effect} */
    br
  );
  return dn(() => {
    var u, l;
    return Sa(() => {
      u = l, l = [], ue(() => {
        Cr(r(...l), e) || (t(e, ...l), u && Cr(r(...u), e) && t(null, ...u));
      });
    }), () => {
      let c = a;
      for (; c !== i && c.parent !== null && c.parent.f & Aa; )
        c = c.parent;
      const f = () => {
        l && Cr(r(...l), e) && t(null, ...l);
      }, p = c.teardown;
      c.teardown = () => {
        f(), p?.();
      };
    };
  }), e;
}
function bs(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Nt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Be(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const u = qr(() => {
      let l = !1;
      const c = t.s;
      for (const f in c)
        c[f] !== a[f] && (a[f] = c[f], l = !0);
      return l && i++, i;
    });
    n = () => o(u);
  }
  r.b.length && Ha(() => {
    Bn(t, n), Xr(r.b);
  }), Le(() => {
    const i = ue(() => r.m.map(Pa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Le(() => {
    Bn(t, n), Xr(r.a);
  });
}
function Bn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) o(r);
  t();
}
const _s = {
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
function ys(e, t, r) {
  return new Proxy({ props: e, exclude: t }, _s);
}
const xs = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (qt(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      qt(i) && (i = i());
      const a = Wr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (qt(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Wr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === mn || t === oi) return !1;
    for (let r of e.props)
      if (qt(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (qt(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function Es(...e) {
  return new Proxy({ props: e }, xs);
}
function N(e, t, r, n) {
  var i = !ni || (r & Ba) !== 0, a = (r & Oa) !== 0, u = (r & Na) !== 0, l = (
    /** @type {V} */
    n
  ), c = !0, f = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), p = () => u && i ? (f ??= qr(
    /** @type {() => V} */
    n
  ), o(f)) : (c && (c = !1, l = u ? ue(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), l);
  let _;
  if (a) {
    var x = mn in e || oi in e;
    _ = Wr(e, t)?.set ?? (x && t in e ? (E) => e[t] = E : void 0);
  }
  var S, v = !1;
  a ? [S, v] = Xa(() => (
    /** @type {V} */
    e[t]
  )) : S = /** @type {V} */
  e[t], S === void 0 && n !== void 0 && (S = p(), _ && (i && Ia(), _(S)));
  var A;
  if (i ? A = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E === void 0 ? p() : (c = !0, E);
  } : A = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E !== void 0 && (l = /** @type {V} */
    void 0), E === void 0 ? l : E;
  }, i && (r & Ma) === 0)
    return A;
  if (_) {
    var P = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(E, w) {
        return arguments.length > 0 ? ((!i || !w || P || v) && _(w ? A() : E), E) : A();
      })
    );
  }
  var d = !1, y = ((r & La) !== 0 ? qr : ii)(() => (d = !1, A()));
  a && o(y);
  var H = (
    /** @type {Effect} */
    br
  );
  return (
    /** @type {() => V} */
    (function(E, w) {
      if (arguments.length > 0) {
        const B = w ? o(y) : i && a ? Qt(E) : E;
        return b(y, B), d = !0, l !== void 0 && (l = B), E;
      }
      return Ca && d || (H.f & ai) !== 0 ? y.v : o(y);
    })
  );
}
const ws = [
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
ws.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: Ln[t][r],
    secondary: Ln[t][n]
  }
}), {});
function Ts(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Rr, Nn;
function Ss() {
  if (Nn) return Rr;
  Nn = 1;
  var e = function(y) {
    return t(y) && !r(y);
  };
  function t(d) {
    return !!d && typeof d == "object";
  }
  function r(d) {
    var y = Object.prototype.toString.call(d);
    return y === "[object RegExp]" || y === "[object Date]" || a(d);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(d) {
    return d.$$typeof === i;
  }
  function u(d) {
    return Array.isArray(d) ? [] : {};
  }
  function l(d, y) {
    return y.clone !== !1 && y.isMergeableObject(d) ? A(u(d), d, y) : d;
  }
  function c(d, y, H) {
    return d.concat(y).map(function(E) {
      return l(E, H);
    });
  }
  function f(d, y) {
    if (!y.customMerge)
      return A;
    var H = y.customMerge(d);
    return typeof H == "function" ? H : A;
  }
  function p(d) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(d).filter(function(y) {
      return Object.propertyIsEnumerable.call(d, y);
    }) : [];
  }
  function _(d) {
    return Object.keys(d).concat(p(d));
  }
  function x(d, y) {
    try {
      return y in d;
    } catch {
      return !1;
    }
  }
  function S(d, y) {
    return x(d, y) && !(Object.hasOwnProperty.call(d, y) && Object.propertyIsEnumerable.call(d, y));
  }
  function v(d, y, H) {
    var E = {};
    return H.isMergeableObject(d) && _(d).forEach(function(w) {
      E[w] = l(d[w], H);
    }), _(y).forEach(function(w) {
      S(d, w) || (x(d, w) && H.isMergeableObject(y[w]) ? E[w] = f(w, H)(d[w], y[w], H) : E[w] = l(y[w], H));
    }), E;
  }
  function A(d, y, H) {
    H = H || {}, H.arrayMerge = H.arrayMerge || c, H.isMergeableObject = H.isMergeableObject || e, H.cloneUnlessOtherwiseSpecified = l;
    var E = Array.isArray(y), w = Array.isArray(d), B = E === w;
    return B ? E ? H.arrayMerge(d, y, H) : v(d, y, H) : l(y, H);
  }
  A.all = function(y, H) {
    if (!Array.isArray(y))
      throw new Error("first argument should be an array");
    return y.reduce(function(E, w) {
      return A(E, w, H);
    }, {});
  };
  var P = A;
  return Rr = P, Rr;
}
var As = Ss();
const Hs = /* @__PURE__ */ Ts(As);
var Kr = function(e, t) {
  return Kr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Kr(e, t);
};
function wr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Kr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var j = function() {
  return j = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, j.apply(this, arguments);
};
function Ps(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function Dr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function kr(e, t) {
  var r = t && t.cache ? t.cache : Cs, n = t && t.serializer ? t.serializer : Ls, i = t && t.strategy ? t.strategy : Os;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function Is(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Ms(e, t, r, n) {
  var i = Is(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function bi(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function _i(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function Os(e, t) {
  var r = e.length === 1 ? Ms : bi;
  return _i(e, this, r, t.cache.create(), t.serializer);
}
function Bs(e, t) {
  return _i(e, this, bi, t.cache.create(), t.serializer);
}
var Ls = function() {
  return JSON.stringify(arguments);
}, Ns = (
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
), Cs = {
  create: function() {
    return new Ns();
  }
}, Ur = {
  variadic: Bs
}, k;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(k || (k = {}));
var J;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(J || (J = {}));
var Dt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(Dt || (Dt = {}));
function Cn(e) {
  return e.type === J.literal;
}
function Rs(e) {
  return e.type === J.argument;
}
function yi(e) {
  return e.type === J.number;
}
function xi(e) {
  return e.type === J.date;
}
function Ei(e) {
  return e.type === J.time;
}
function wi(e) {
  return e.type === J.select;
}
function Ti(e) {
  return e.type === J.plural;
}
function Ds(e) {
  return e.type === J.pound;
}
function Si(e) {
  return e.type === J.tag;
}
function Ai(e) {
  return !!(e && typeof e == "object" && e.type === Dt.number);
}
function $r(e) {
  return !!(e && typeof e == "object" && e.type === Dt.dateTime);
}
var Hi = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, ks = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function Us(e) {
  var t = {};
  return e.replace(ks, function(r) {
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
var Fs = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Gs(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Fs).filter(function(x) {
    return x.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], u = a.split("/");
    if (u.length === 0)
      throw new Error("Invalid number skeleton");
    for (var l = u[0], c = u.slice(1), f = 0, p = c; f < p.length; f++) {
      var _ = p[f];
      if (_.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: l, options: c });
  }
  return r;
}
function js(e) {
  return e.replace(/^(.*?)-/, "");
}
var Rn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, Pi = /^(@+)?(\+|#+)?[rs]?$/g, Vs = /(\*)(0+)|(#+)(0+)|(0+)/g, Ii = /^(0+)$/;
function Dn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(Pi, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function Mi(e) {
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
function zs(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !Ii.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function kn(e) {
  var t = {}, r = Mi(e);
  return r || t;
}
function Xs(e) {
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
        t.style = "unit", t.unit = js(i.options[0]);
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
        t = j(j(j({}, t), { notation: "scientific" }), i.options.reduce(function(c, f) {
          return j(j({}, c), kn(f));
        }, {}));
        continue;
      case "engineering":
        t = j(j(j({}, t), { notation: "engineering" }), i.options.reduce(function(c, f) {
          return j(j({}, c), kn(f));
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
        i.options[0].replace(Vs, function(c, f, p, _, x, S) {
          if (f)
            t.minimumIntegerDigits = p.length;
          else {
            if (_ && x)
              throw new Error("We currently do not support maximum integer digits");
            if (S)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (Ii.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (Rn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(Rn, function(c, f, p, _, x, S) {
        return p === "*" ? t.minimumFractionDigits = f.length : _ && _[0] === "#" ? t.maximumFractionDigits = _.length : x && S ? (t.minimumFractionDigits = x.length, t.maximumFractionDigits = x.length + S.length) : (t.minimumFractionDigits = f.length, t.maximumFractionDigits = f.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = j(j({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = j(j({}, t), Dn(a)));
      continue;
    }
    if (Pi.test(i.stem)) {
      t = j(j({}, t), Dn(i.stem));
      continue;
    }
    var u = Mi(i.stem);
    u && (t = j(j({}, t), u));
    var l = zs(i.stem);
    l && (t = j(j({}, t), l));
  }
  return t;
}
var lr = {
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
function qs(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var u = 1 + (a & 1), l = a < 2 ? 1 : 3 + (a >> 1), c = "a", f = Ws(t);
      for ((f == "H" || f == "k") && (l = 0); l-- > 0; )
        r += c;
      for (; u-- > 0; )
        r = f + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Ws(e) {
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
  var i = lr[n || ""] || lr[r || ""] || lr["".concat(r, "-001")] || lr["001"];
  return i[0];
}
var Fr, Ys = new RegExp("^".concat(Hi.source, "*")), Zs = new RegExp("".concat(Hi.source, "*$"));
function F(e, t) {
  return { start: e, end: t };
}
var Js = !!String.prototype.startsWith && "_a".startsWith("a", 1), Qs = !!String.fromCodePoint, Ks = !!Object.fromEntries, $s = !!String.prototype.codePointAt, eo = !!String.prototype.trimStart, to = !!String.prototype.trimEnd, ro = !!Number.isSafeInteger, no = ro ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, en = !0;
try {
  var io = Bi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  en = ((Fr = io.exec("a")) === null || Fr === void 0 ? void 0 : Fr[0]) === "a";
} catch {
  en = !1;
}
var Un = Js ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), tn = Qs ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", i = t.length, a = 0, u; i > a; ) {
      if (u = t[a++], u > 1114111)
        throw RangeError(u + " is not a valid code point");
      n += u < 65536 ? String.fromCharCode(u) : String.fromCharCode(((u -= 65536) >> 10) + 55296, u % 1024 + 56320);
    }
    return n;
  }
), Fn = (
  // native
  Ks ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], u = a[0], l = a[1];
        r[u] = l;
      }
      return r;
    }
  )
), Oi = $s ? (
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
), ao = eo ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ys, "");
  }
), so = to ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Zs, "");
  }
);
function Bi(e, t) {
  return new RegExp(e, t);
}
var rn;
if (en) {
  var Gn = Bi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  rn = function(t, r) {
    var n;
    Gn.lastIndex = r;
    var i = Gn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  rn = function(t, r) {
    for (var n = []; ; ) {
      var i = Oi(t, r);
      if (i === void 0 || Li(i) || fo(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return tn.apply(void 0, n);
  };
var oo = (
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
          var u = this.parseArgument(t, n);
          if (u.err)
            return u;
          i.push(u.val);
        } else {
          if (a === 125 && t > 0)
            break;
          if (a === 35 && (r === "plural" || r === "selectordinal")) {
            var l = this.clonePosition();
            this.bump(), i.push({
              type: J.pound,
              location: F(l, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(k.UNMATCHED_CLOSING_TAG, F(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && nn(this.peek() || 0)) {
            var u = this.parseTag(t, r);
            if (u.err)
              return u;
            i.push(u.val);
          } else {
            var u = this.parseLiteral(t, r);
            if (u.err)
              return u;
            i.push(u.val);
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
            type: J.literal,
            value: "<".concat(i, "/>"),
            location: F(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var u = a.val, l = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !nn(this.char()))
            return this.error(k.INVALID_TAG, F(l, this.clonePosition()));
          var c = this.clonePosition(), f = this.parseTagName();
          return i !== f ? this.error(k.UNMATCHED_CLOSING_TAG, F(c, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: J.tag,
              value: i,
              children: u,
              location: F(n, this.clonePosition())
            },
            err: null
          } : this.error(k.INVALID_TAG, F(l, this.clonePosition())));
        } else
          return this.error(k.UNCLOSED_TAG, F(n, this.clonePosition()));
      } else
        return this.error(k.INVALID_TAG, F(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && uo(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), i = ""; ; ) {
        var a = this.tryParseQuote(r);
        if (a) {
          i += a;
          continue;
        }
        var u = this.tryParseUnquoted(t, r);
        if (u) {
          i += u;
          continue;
        }
        var l = this.tryParseLeftAngleBracket();
        if (l) {
          i += l;
          continue;
        }
        break;
      }
      var c = F(n, this.clonePosition());
      return {
        val: { type: J.literal, value: i, location: c },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !lo(this.peek() || 0)) ? (this.bump(), "<") : null;
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
        return this.error(k.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(k.EMPTY_ARGUMENT, F(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(k.MALFORMED_ARGUMENT, F(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(k.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: J.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: F(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(k.EXPECT_ARGUMENT_CLOSING_BRACE, F(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(k.MALFORMED_ARGUMENT, F(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = rn(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), u = F(t, a);
      return { value: n, location: u };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, u = this.clonePosition(), l = this.parseIdentifierIfPossible().value, c = this.clonePosition();
      switch (l) {
        case "":
          return this.error(k.EXPECT_ARGUMENT_TYPE, F(u, c));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var f = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var p = this.clonePosition(), _ = this.parseSimpleArgStyleIfPossible();
            if (_.err)
              return _;
            var x = so(_.val);
            if (x.length === 0)
              return this.error(k.EXPECT_ARGUMENT_STYLE, F(this.clonePosition(), this.clonePosition()));
            var S = F(p, this.clonePosition());
            f = { style: x, styleLocation: S };
          }
          var v = this.tryParseArgumentClose(i);
          if (v.err)
            return v;
          var A = F(i, this.clonePosition());
          if (f && Un(f?.style, "::", 0)) {
            var P = ao(f.style.slice(2));
            if (l === "number") {
              var _ = this.parseNumberSkeletonFromString(P, f.styleLocation);
              return _.err ? _ : {
                val: { type: J.number, value: n, location: A, style: _.val },
                err: null
              };
            } else {
              if (P.length === 0)
                return this.error(k.EXPECT_DATE_TIME_SKELETON, A);
              var d = P;
              this.locale && (d = qs(P, this.locale));
              var x = {
                type: Dt.dateTime,
                pattern: d,
                location: f.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? Us(d) : {}
              }, y = l === "date" ? J.date : J.time;
              return {
                val: { type: y, value: n, location: A, style: x },
                err: null
              };
            }
          }
          return {
            val: {
              type: l === "number" ? J.number : l === "date" ? J.date : J.time,
              value: n,
              location: A,
              style: (a = f?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var H = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(k.EXPECT_SELECT_ARGUMENT_OPTIONS, F(H, j({}, H)));
          this.bumpSpace();
          var E = this.parseIdentifierIfPossible(), w = 0;
          if (l !== "select" && E.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(k.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, F(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var _ = this.tryParseDecimalInteger(k.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, k.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (_.err)
              return _;
            this.bumpSpace(), E = this.parseIdentifierIfPossible(), w = _.val;
          }
          var B = this.tryParsePluralOrSelectOptions(t, l, r, E);
          if (B.err)
            return B;
          var v = this.tryParseArgumentClose(i);
          if (v.err)
            return v;
          var M = F(i, this.clonePosition());
          return l === "select" ? {
            val: {
              type: J.select,
              value: n,
              options: Fn(B.val),
              location: M
            },
            err: null
          } : {
            val: {
              type: J.plural,
              value: n,
              options: Fn(B.val),
              offset: w,
              pluralType: l === "plural" ? "cardinal" : "ordinal",
              location: M
            },
            err: null
          };
        }
        default:
          return this.error(k.INVALID_ARGUMENT_TYPE, F(u, c));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(k.EXPECT_ARGUMENT_CLOSING_BRACE, F(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(k.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, F(i, this.clonePosition()));
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
        return this.error(k.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: Dt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Xs(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, u = !1, l = [], c = /* @__PURE__ */ new Set(), f = i.value, p = i.location; ; ) {
        if (f.length === 0) {
          var _ = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var x = this.tryParseDecimalInteger(k.EXPECT_PLURAL_ARGUMENT_SELECTOR, k.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (x.err)
              return x;
            p = F(_, this.clonePosition()), f = this.message.slice(_.offset, this.offset());
          } else
            break;
        }
        if (c.has(f))
          return this.error(r === "select" ? k.DUPLICATE_SELECT_ARGUMENT_SELECTOR : k.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, p);
        f === "other" && (u = !0), this.bumpSpace();
        var S = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? k.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : k.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, F(this.clonePosition(), this.clonePosition()));
        var v = this.parseMessage(t + 1, r, n);
        if (v.err)
          return v;
        var A = this.tryParseArgumentClose(S);
        if (A.err)
          return A;
        l.push([
          f,
          {
            value: v.val,
            location: F(S, this.clonePosition())
          }
        ]), c.add(f), this.bumpSpace(), a = this.parseIdentifierIfPossible(), f = a.value, p = a.location;
      }
      return l.length === 0 ? this.error(r === "select" ? k.EXPECT_SELECT_ARGUMENT_SELECTOR : k.EXPECT_PLURAL_ARGUMENT_SELECTOR, F(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !u ? this.error(k.MISSING_OTHER_CLAUSE, F(this.clonePosition(), this.clonePosition())) : { val: l, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var a = !1, u = 0; !this.isEOF(); ) {
        var l = this.char();
        if (l >= 48 && l <= 57)
          a = !0, u = u * 10 + (l - 48), this.bump();
        else
          break;
      }
      var c = F(i, this.clonePosition());
      return a ? (u *= n, no(u) ? { val: u, err: null } : this.error(r, c)) : this.error(t, c);
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
      var r = Oi(this.message, t);
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
      for (; !this.isEOF() && Li(this.char()); )
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
function lo(e) {
  return nn(e) || e === 47;
}
function uo(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function Li(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function fo(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function an(e) {
  e.forEach(function(t) {
    if (delete t.location, wi(t) || Ti(t))
      for (var r in t.options)
        delete t.options[r].location, an(t.options[r].value);
    else yi(t) && Ai(t.style) || (xi(t) || Ei(t)) && $r(t.style) ? delete t.style.location : Si(t) && an(t.children);
  });
}
function co(e, t) {
  t === void 0 && (t = {}), t = j({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new oo(e, t).parse();
  if (r.err) {
    var n = SyntaxError(k[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || an(r.val), r.val;
}
var kt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(kt || (kt = {}));
var Tr = (
  /** @class */
  (function(e) {
    wr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), jn = (
  /** @class */
  (function(e) {
    wr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), kt.INVALID_VALUE, a) || this;
    }
    return t;
  })(Tr)
), ho = (
  /** @class */
  (function(e) {
    wr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), kt.INVALID_VALUE, i) || this;
    }
    return t;
  })(Tr)
), po = (
  /** @class */
  (function(e) {
    wr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), kt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Tr)
), Ee;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(Ee || (Ee = {}));
function mo(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== Ee.literal || r.type !== Ee.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function go(e) {
  return typeof e == "function";
}
function cr(e, t, r, n, i, a, u) {
  if (e.length === 1 && Cn(e[0]))
    return [
      {
        type: Ee.literal,
        value: e[0].value
      }
    ];
  for (var l = [], c = 0, f = e; c < f.length; c++) {
    var p = f[c];
    if (Cn(p)) {
      l.push({
        type: Ee.literal,
        value: p.value
      });
      continue;
    }
    if (Ds(p)) {
      typeof a == "number" && l.push({
        type: Ee.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var _ = p.value;
    if (!(i && _ in i))
      throw new po(_, u);
    var x = i[_];
    if (Rs(p)) {
      (!x || typeof x == "string" || typeof x == "number") && (x = typeof x == "string" || typeof x == "number" ? String(x) : ""), l.push({
        type: typeof x == "string" ? Ee.literal : Ee.object,
        value: x
      });
      continue;
    }
    if (xi(p)) {
      var S = typeof p.style == "string" ? n.date[p.style] : $r(p.style) ? p.style.parsedOptions : void 0;
      l.push({
        type: Ee.literal,
        value: r.getDateTimeFormat(t, S).format(x)
      });
      continue;
    }
    if (Ei(p)) {
      var S = typeof p.style == "string" ? n.time[p.style] : $r(p.style) ? p.style.parsedOptions : n.time.medium;
      l.push({
        type: Ee.literal,
        value: r.getDateTimeFormat(t, S).format(x)
      });
      continue;
    }
    if (yi(p)) {
      var S = typeof p.style == "string" ? n.number[p.style] : Ai(p.style) ? p.style.parsedOptions : void 0;
      S && S.scale && (x = x * (S.scale || 1)), l.push({
        type: Ee.literal,
        value: r.getNumberFormat(t, S).format(x)
      });
      continue;
    }
    if (Si(p)) {
      var v = p.children, A = p.value, P = i[A];
      if (!go(P))
        throw new ho(A, "function", u);
      var d = cr(v, t, r, n, i, a), y = P(d.map(function(w) {
        return w.value;
      }));
      Array.isArray(y) || (y = [y]), l.push.apply(l, y.map(function(w) {
        return {
          type: typeof w == "string" ? Ee.literal : Ee.object,
          value: w
        };
      }));
    }
    if (wi(p)) {
      var H = p.options[x] || p.options.other;
      if (!H)
        throw new jn(p.value, x, Object.keys(p.options), u);
      l.push.apply(l, cr(H.value, t, r, n, i));
      continue;
    }
    if (Ti(p)) {
      var H = p.options["=".concat(x)];
      if (!H) {
        if (!Intl.PluralRules)
          throw new Tr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, kt.MISSING_INTL_API, u);
        var E = r.getPluralRules(t, { type: p.pluralType }).select(x - (p.offset || 0));
        H = p.options[E] || p.options.other;
      }
      if (!H)
        throw new jn(p.value, x, Object.keys(p.options), u);
      l.push.apply(l, cr(H.value, t, r, n, i, x - (p.offset || 0)));
      continue;
    }
  }
  return mo(l);
}
function vo(e, t) {
  return t ? j(j(j({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = j(j({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function bo(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = vo(e[n], t[n]), r;
  }, j({}, e)) : e;
}
function Gr(e) {
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
function _o(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Dr([void 0], r, !1)))();
    }, {
      cache: Gr(e.number),
      strategy: Ur.variadic
    }),
    getDateTimeFormat: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Dr([void 0], r, !1)))();
    }, {
      cache: Gr(e.dateTime),
      strategy: Ur.variadic
    }),
    getPluralRules: kr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Dr([void 0], r, !1)))();
    }, {
      cache: Gr(e.pluralRules),
      strategy: Ur.variadic
    })
  };
}
var yo = (
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
        var p = f.reduce(function(_, x) {
          return !_.length || x.type !== Ee.literal || typeof _[_.length - 1] != "string" ? _.push(x.value) : _[_.length - 1] += x.value, _;
        }, []);
        return p.length <= 1 ? p[0] || "" : p;
      }, this.formatToParts = function(c) {
        return cr(a.ast, a.locales, a.formatters, a.formats, c, void 0, a.message);
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
        var u = i || {};
        u.formatters;
        var l = Ps(u, ["formatters"]);
        this.ast = e.__parse(t, j(j({}, l), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = bo(e.formats, n), this.formatters = i && i.formatters || _o(this.formatterCache);
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
    }, e.__parse = co, e.formats = {
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
function xo(e, t) {
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
const ot = {}, Eo = (e, t, r) => r && (t in ot || (ot[t] = {}), e in ot[t] || (ot[t][e] = r), r), Ni = (e, t) => {
  if (t == null)
    return;
  if (t in ot && e in ot[t])
    return ot[t][e];
  const r = Sr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = To(i, e);
    if (a)
      return Eo(e, t, a);
  }
};
let vn;
const rr = tr({});
function wo(e) {
  return vn[e] || null;
}
function Ci(e) {
  return e in vn;
}
function To(e, t) {
  if (!Ci(e))
    return null;
  const r = wo(e);
  return xo(r, t);
}
function So(e) {
  if (e == null)
    return;
  const t = Sr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Ci(n))
      return n;
  }
}
function Ao(e, ...t) {
  delete ot[e], rr.update((r) => (r[e] = Hs.all([r[e] || {}, ...t]), r));
}
Ft(
  [rr],
  ([e]) => Object.keys(e)
);
rr.subscribe((e) => vn = e);
const hr = {};
function Ho(e, t) {
  hr[e].delete(t), hr[e].size === 0 && delete hr[e];
}
function Ri(e) {
  return hr[e];
}
function Po(e) {
  return Sr(e).map((t) => {
    const r = Ri(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function sn(e) {
  return e == null ? !1 : Sr(e).some(
    (t) => {
      var r;
      return (r = Ri(t)) == null ? void 0 : r.size;
    }
  );
}
function Io(e, t) {
  return Promise.all(
    t.map((n) => (Ho(e, n), n().then((i) => i.default || i)))
  ).then((n) => Ao(e, ...n));
}
const Yt = {};
function Di(e) {
  if (!sn(e))
    return e in Yt ? Yt[e] : Promise.resolve();
  const t = Po(e);
  return Yt[e] = Promise.all(
    t.map(
      ([r, n]) => Io(r, n)
    )
  ).then(() => {
    if (sn(e))
      return Di(e);
    delete Yt[e];
  }), Yt[e];
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
}, Bo = Oo;
function Ut() {
  return Bo;
}
const jr = tr(!1);
var Lo = Object.defineProperty, No = Object.defineProperties, Co = Object.getOwnPropertyDescriptors, Vn = Object.getOwnPropertySymbols, Ro = Object.prototype.hasOwnProperty, Do = Object.prototype.propertyIsEnumerable, zn = (e, t, r) => t in e ? Lo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, ko = (e, t) => {
  for (var r in t || (t = {}))
    Ro.call(t, r) && zn(e, r, t[r]);
  if (Vn)
    for (var r of Vn(t))
      Do.call(t, r) && zn(e, r, t[r]);
  return e;
}, Uo = (e, t) => No(e, Co(t));
let on;
const gr = tr(null);
function Xn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Sr(e, t = Ut().fallbackLocale) {
  const r = Xn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Xn(t)])] : r;
}
function Tt() {
  return on ?? void 0;
}
gr.subscribe((e) => {
  on = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const Fo = (e) => {
  if (e && So(e) && sn(e)) {
    const { loadingDelay: t } = Ut();
    let r;
    return typeof window < "u" && Tt() != null && t ? r = window.setTimeout(
      () => jr.set(!0),
      t
    ) : jr.set(!0), Di(e).then(() => {
      gr.set(e);
    }).finally(() => {
      clearTimeout(r), jr.set(!1);
    });
  }
  return gr.set(e);
}, Gt = Uo(ko({}, gr), {
  set: Fo
}), Ar = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Go = Object.defineProperty, vr = Object.getOwnPropertySymbols, ki = Object.prototype.hasOwnProperty, Ui = Object.prototype.propertyIsEnumerable, qn = (e, t, r) => t in e ? Go(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, bn = (e, t) => {
  for (var r in t || (t = {}))
    ki.call(t, r) && qn(e, r, t[r]);
  if (vr)
    for (var r of vr(t))
      Ui.call(t, r) && qn(e, r, t[r]);
  return e;
}, jt = (e, t) => {
  var r = {};
  for (var n in e)
    ki.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && vr)
    for (var n of vr(e))
      t.indexOf(n) < 0 && Ui.call(e, n) && (r[n] = e[n]);
  return r;
};
const Kt = (e, t) => {
  const { formats: r } = Ut();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, jo = Ar(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Kt("number", n)), new Intl.NumberFormat(r, i);
  }
), Vo = Ar(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Kt("date", n) : Object.keys(i).length === 0 && (i = Kt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), zo = Ar(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = jt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Kt("time", n) : Object.keys(i).length === 0 && (i = Kt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Xo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return jo(bn({ locale: r }, n));
}, qo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return Vo(bn({ locale: r }, n));
}, Wo = (e = {}) => {
  var t = e, {
    locale: r = Tt()
  } = t, n = jt(t, [
    "locale"
  ]);
  return zo(bn({ locale: r }, n));
}, Yo = Ar(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Tt()) => new yo(e, t, Ut().formats, {
    ignoreTag: Ut().ignoreTag
  })
), Zo = (e, t = {}) => {
  var r, n, i, a;
  let u = t;
  typeof e == "object" && (u = e, e = u.id);
  const {
    values: l,
    locale: c = Tt(),
    default: f
  } = u;
  if (c == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let p = Ni(e, c);
  if (!p)
    p = (a = (i = (n = (r = Ut()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: c, id: e, defaultValue: f })) != null ? i : f) != null ? a : e;
  else if (typeof p != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof p}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), p;
  if (!l)
    return p;
  let _ = p;
  try {
    _ = Yo(p, c).format(l);
  } catch (x) {
    x instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      x.message
    );
  }
  return _;
}, Jo = (e, t) => Wo(t).format(e), Qo = (e, t) => qo(t).format(e), Ko = (e, t) => Xo(t).format(e), $o = (e, t = Tt()) => Ni(e, t);
Ft([Gt, rr], () => Zo);
Ft([Gt], () => Jo);
Ft([Gt], () => Qo);
Ft([Gt], () => Ko);
Ft([Gt, rr], () => $o);
const el = "__i18n__", tl = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], rl = [
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
function nl(e) {
  return typeof e == "string" && e.includes(el);
}
class il {
  load_component;
  #t = z(Qt({}));
  get shared() {
    return o(this.#t);
  }
  set shared(t) {
    b(this.#t, t, !0);
  }
  #r = z(Qt({}));
  get props() {
    return o(this.#r);
  }
  set props(t) {
    b(this.#r, t, !0);
  }
  #e = z((t) => t);
  get i18n() {
    return o(this.#e);
  }
  set i18n(t) {
    b(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = rl;
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
    for (const n of tl)
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
      ), ue(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Gt.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [a, u] = n.split("."), l = this.i18n(i);
        a === "shared" ? this.shared[u] = l : this.props[u] = l;
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
      const n = t[r], i = nl(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
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
Ra();
var al = /* @__PURE__ */ fi('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Wn = /* @__PURE__ */ de("<!> <!>", 1), sl = /* @__PURE__ */ de('<div class="placeholder svelte-1stq1b1"></div>');
function ol(e, t) {
  xr(t, !1);
  let r = N(t, "height", 8, void 0), n = N(t, "min_height", 8, void 0), i = N(t, "max_height", 8, void 0), a = N(t, "width", 8, void 0), u = N(t, "elem_id", 8, ""), l = N(t, "elem_classes", 24, () => []), c = N(t, "variant", 8, "solid"), f = N(t, "border_mode", 8, "base"), p = N(t, "padding", 8, !0), _ = N(t, "type", 8, "normal"), x = N(t, "test_id", 8, void 0), S = N(t, "explicit_call", 8, !1), v = N(t, "container", 8, !0), A = N(t, "visible", 8, !0), P = N(t, "allow_overflow", 8, !0), d = N(t, "overflow_behavior", 8, "auto"), y = N(t, "scale", 8, null), H = N(t, "min_width", 8, 0), E = N(t, "flex", 12, !1), w = N(t, "resizable", 8, !1), B = N(t, "rtl", 8, !1), M = N(t, "fullscreen", 12, !1), L = N(t, "label", 8, void 0), R = yt(M()), C = yt(), G = _() === "fieldset" ? "fieldset" : "div", se = yt(0), q = yt(0), W = yt(null);
  function We(oe) {
    M() && oe.key === "Escape" && M(!1);
  }
  const Pe = (oe) => {
    if (oe !== void 0) {
      if (typeof oe == "number")
        return oe + "px";
      if (typeof oe == "string")
        return oe;
    }
  }, Ne = (oe) => {
    let Ie = oe.clientY;
    const me = (ee) => {
      const fe = ee.clientY - Ie;
      Ie = ee.clientY, ka(C, o(C).style.height = `${o(C).offsetHeight + fe}px`);
    }, we = () => {
      window.removeEventListener("mousemove", me), window.removeEventListener("mouseup", we);
    };
    window.addEventListener("mousemove", me), window.addEventListener("mouseup", we);
  };
  wn(
    () => (Be(M()), o(R), o(C)),
    () => {
      M() !== o(R) && (b(R, M()), M() ? (b(W, o(C).getBoundingClientRect()), b(se, o(C).offsetHeight), b(q, o(C).offsetWidth), window.addEventListener("keydown", We)) : (b(W, null), window.removeEventListener("keydown", We)));
    }
  ), wn(() => Be(A()), () => {
    A() || E(!1);
  }), Da(), bs();
  var ft = Lt(), pe = xe(ft);
  {
    var St = (oe) => {
      var Ie = Wn(), me = xe(Ie);
      as(me, () => G, !1, (fe, K) => {
        mr(fe, (Te) => b(C, Te), () => o(C)), vs(
          fe,
          (Te, ge) => ({
            "data-testid": x(),
            id: u(),
            class: `block ${Te ?? ""}`,
            dir: B() ? "rtl" : "ltr",
            "aria-label": L(),
            style: "",
            [Jt]: {
              hidden: A() === "hidden",
              padded: p(),
              flex: E(),
              border_focus: f() === "focus",
              border_contrast: f() === "contrast",
              "hide-container": !S() && !v(),
              fullscreen: M(),
              animating: M() && o(W) !== null,
              "auto-margin": y() === null
            },
            [Bt]: ge
          }),
          [
            () => (Be(l()), ue(() => l()?.join(" ") || "")),
            () => ({
              height: (Be(M()), Be(r()), ue(() => M() ? void 0 : Pe(r()))),
              "min-height": (Be(M()), Be(n()), ue(() => M() ? void 0 : Pe(n()))),
              "max-height": (Be(M()), Be(i()), ue(() => M() ? void 0 : Pe(i()))),
              "--start-top": (o(W), ue(() => o(W) ? `${o(W).top}px` : "0px")),
              "--start-left": (o(W), ue(() => o(W) ? `${o(W).left}px` : "0px")),
              "--start-width": (o(W), ue(() => o(W) ? `${o(W).width}px` : "0px")),
              "--start-height": (o(W), ue(() => o(W) ? `${o(W).height}px` : "0px")),
              width: (Be(M()), Be(a()), ue(() => M() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : Pe(a()))),
              "border-style": c(),
              overflow: P() ? d() : "hidden",
              "flex-grow": y(),
              "min-width": `calc(min(${H()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Ge = Wn(), Ye = xe(Ge);
        Jr(Ye, t, "default", {});
        var Ce = Z(Ye, 2);
        {
          var Ze = (Te) => {
            var ge = al();
            Ot("mousedown", ge, Ne), U(Te, ge);
          };
          $(Ce, (Te) => {
            w() && Te(Ze);
          });
        }
        U(K, Ge);
      });
      var we = Z(me, 2);
      {
        var ee = (fe) => {
          var K = sl();
          let Ge;
          te(() => Ge = Fe(K, "", Ge, {
            height: o(se) + "px",
            width: o(q) + "px"
          })), U(fe, K);
        };
        $(we, (fe) => {
          M() && fe(ee);
        });
      }
      U(oe, Ie);
    };
    $(pe, (oe) => {
      (A() === !0 || A() === "hidden") && oe(St);
    });
  }
  U(e, ft), yr();
}
var ll = /* @__PURE__ */ de('<span class="svelte-vvirtv"> </span>'), ul = /* @__PURE__ */ de("<button><!> <div><!> <!></div></button>");
function Yn(e, t) {
  let r = N(t, "label", 3, ""), n = N(t, "show_label", 3, !1), i = N(t, "pending", 3, !1), a = N(t, "size", 3, "small"), u = N(t, "padded", 3, !0), l = N(t, "highlight", 3, !1), c = N(t, "disabled", 3, !1), f = N(t, "hasPopup", 3, !1), p = N(t, "color", 3, "var(--block-label-text-color)"), _ = N(t, "transparent", 3, !1), x = N(t, "background", 3, "var(--block-background-fill)"), S = N(t, "border", 3, "transparent"), v = ke(() => l() ? "var(--color-accent)" : p());
  var A = ul();
  let P, d;
  var y = ae(A);
  {
    var H = (R) => {
      var C = ll(), G = ae(C);
      te(() => ye(G, r())), U(R, C);
    };
    $(y, (R) => {
      n() && R(H);
    });
  }
  var E = Z(y, 2);
  let w;
  var B = ae(E);
  rs(B, () => t.Icon, (R, C) => {
    C(R, {});
  });
  var M = Z(B, 2);
  {
    var L = (R) => {
      var C = Lt(), G = xe(C);
      Ja(G, () => t.children), U(R, C);
    };
    $(M, (R) => {
      t.children && R(L);
    });
  }
  te(() => {
    P = ut(A, 1, "icon-button svelte-vvirtv", null, P, {
      pending: i(),
      padded: u(),
      highlight: l(),
      transparent: _()
    }), A.disabled = c(), Ct(A, "aria-label", r()), Ct(A, "aria-haspopup", f()), Ct(A, "title", r()), d = Fe(A, "", d, {
      "--border-color": S(),
      color: !c() && o(v) ? o(v) : "var(--block-label-text-color)",
      "--bg-color": c() ? "auto" : x()
    }), w = ut(E, 1, "svelte-vvirtv", null, w, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), st("click", A, function(...R) {
    t.onclick?.apply(this, R);
  }), U(e, A);
}
er(["click"]);
var fl = /* @__PURE__ */ fi('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Zn(e) {
  var t = fl();
  U(e, t);
}
er(["click"]);
function Vr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Jn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function ln(e, t, r, n) {
  if (typeof r == "number" || Jn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), u = e.opts.stiffness * i, l = e.opts.damping * a, c = (u - l) * e.inv_mass, f = (a + c) * e.dt;
    return Math.abs(f) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Jn(r) ? new Date(r.getTime() + f) : r + f);
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
function Qn(e, t = {}) {
  const r = tr(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let u, l, c, f = (
    /** @type {T} */
    e
  ), p = (
    /** @type {T | undefined} */
    e
  ), _ = 1, x = 0, S = !1;
  function v(P, d = {}) {
    p = P;
    const y = c = {};
    return e == null || d.hard || A.stiffness >= 1 && A.damping >= 1 ? (S = !0, u = Ue.now(), f = P, r.set(e = p), Promise.resolve()) : (d.soft && (x = 1 / ((d.soft === !0 ? 0.5 : +d.soft) * 60), _ = 0), l || (u = Ue.now(), S = !1, l = is((H) => {
      if (S)
        return S = !1, l = null, !1;
      _ = Math.min(_ + x, 1);
      const E = Math.min(H - u, 1e3 / 30), w = {
        inv_mass: _,
        opts: A,
        settled: !0,
        dt: E * 60 / 1e3
      }, B = ln(w, f, e, p);
      return u = H, f = /** @type {T} */
      e, r.set(e = /** @type {T} */
      B), w.settled && (l = null), !w.settled;
    })), new Promise((H) => {
      l.promise.then(() => {
        y === c && H();
      });
    }));
  }
  const A = {
    set: v,
    update: (P, d) => v(P(
      /** @type {T} */
      p,
      /** @type {T} */
      e
    ), d),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return A;
}
var cl = /* @__PURE__ */ de('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function hl(e, t) {
  xr(t, !0);
  const r = () => Tn(c, "$top", i), n = () => Tn(f, "$bottom", i), [i, a] = za();
  var u = this && this.__awaiter || function(H, E, w, B) {
    function M(L) {
      return L instanceof w ? L : new w(function(R) {
        R(L);
      });
    }
    return new (w || (w = Promise))(function(L, R) {
      function C(q) {
        try {
          se(B.next(q));
        } catch (W) {
          R(W);
        }
      }
      function G(q) {
        try {
          se(B.throw(q));
        } catch (W) {
          R(W);
        }
      }
      function se(q) {
        q.done ? L(q.value) : M(q.value).then(C, G);
      }
      se((B = B.apply(H, E || [])).next());
    });
  };
  let l = N(t, "margin", 3, !0);
  const c = Qn([0, 0]), f = Qn([0, 0]);
  let p = z(!1);
  function _() {
    return u(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 140]), f.set([-125, -140])]), yield Promise.all([c.set([-125, 140]), f.set([125, -140])]), yield Promise.all([c.set([-125, 0]), f.set([125, -0])]), yield Promise.all([c.set([125, 0]), f.set([-125, 0])]);
    });
  }
  function x() {
    return u(this, void 0, void 0, function* () {
      yield _(), o(p) || x();
    });
  }
  function S() {
    return u(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 0]), f.set([-125, 0])]), x();
    });
  }
  Le(() => (S(), () => {
    b(p, !0);
  }));
  var v = cl();
  let A;
  var P = ae(v), d = ae(P), y = Z(d);
  te(() => {
    A = ut(v, 1, "svelte-m6d381", null, A, { margin: l() }), Fe(d, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Fe(y, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), U(e, v), yr(), a();
}
var dl = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(u) {
      u(a);
    });
  }
  return new (r || (r = Promise))(function(a, u) {
    function l(p) {
      try {
        f(n.next(p));
      } catch (_) {
        u(_);
      }
    }
    function c(p) {
      try {
        f(n.throw(p));
      } catch (_) {
        u(_);
      }
    }
    function f(p) {
      p.done ? a(p.value) : i(p.value).then(l, c);
    }
    f((n = n.apply(e, t || [])).next());
  });
};
let ur = [], zr = !1;
const pl = typeof window < "u", Fi = pl ? window.requestAnimationFrame : (e) => {
};
function ml(e) {
  return dl(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (ur.push(t), !zr) zr = !0;
      else return;
      yield Ua(), Fi(() => {
        let n = [0, 0];
        for (let i = 0; i < ur.length; i++) {
          const u = ur[i].getBoundingClientRect();
          (i === 0 || u.top + window.scrollY <= n[0]) && (n[0] = u.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), zr = !1, ur = [];
      });
    }
  });
}
var gl = /* @__PURE__ */ de('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), vl = /* @__PURE__ */ de('<div class="eta-bar svelte-124hqw6"></div>'), bl = /* @__PURE__ */ de("<!> ", 1), _l = /* @__PURE__ */ de("<!> <!> <!> <!>", 1), yl = /* @__PURE__ */ de('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), xl = /* @__PURE__ */ de('<p class="loading svelte-124hqw6"> </p> <!>', 1), El = /* @__PURE__ */ de("<!> <div><!> <!></div> <!> <!>", 1), wl = /* @__PURE__ */ de('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), Tl = /* @__PURE__ */ de("<div> <!> </div>"), Sl = /* @__PURE__ */ de('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function Al(e, t) {
  xr(t, !0);
  let r = N(t, "eta", 3, null), n = N(t, "scroll_to_output", 3, !1), i = N(t, "timer", 3, !0), a = N(t, "show_progress", 3, "full"), u = N(t, "message", 3, null), l = N(t, "progress", 3, null), c = N(t, "variant", 3, "default"), f = N(t, "loading_text", 3, "Loading..."), p = N(t, "absolute", 3, !0), _ = N(t, "translucent", 3, !1), x = N(t, "border", 3, !1), S = N(t, "validation_error", 7, null), v = N(t, "show_validation_error", 3, !0), A = N(t, "type", 3, null), P = N(t, "used_cache", 3, null), d = N(t, "cache_duration", 3, null), y = N(t, "avg_time", 3, null), H, E = !1, w = z(0), B = z(null), M = z(null), L = z(!1), R = z(null), C = z(!1), G = z(!1), se = z(null), q = z(null), W = z("from cache"), We = z(!1), Pe = null, Ne = null;
  const ft = ke(() => !(v() && S()) && (A() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let pe = z(0);
  const St = ke(() => o(M) === null || o(M) <= 0 || !o(pe) ? 0 : Math.min(o(pe) / o(M), 1)), oe = ke(() => o(pe).toFixed(1));
  let Ie = ke(() => l() == null), me = ke(() => r() !== null && r() !== void 0 ? r() : o(B));
  function we() {
    Fi(() => {
      b(pe, (performance.now() - o(w)) / 1e3), E && we();
    });
  }
  let ee = ke(() => {
    let X = null;
    l() != null ? X = l().map((ne) => {
      if (ne.index != null && ne.length != null)
        return ne.index / ne.length;
      if (ne.progress != null)
        return ne.progress;
    }) : X = null;
    let re, ce = "";
    return X ? (re = X[X.length - 1], re === 0 ? ce = "0" : ce = "150ms") : re = void 0, {
      progress_level: X,
      last_progress_level: re,
      progress_bar_transition: ce
    };
  });
  function fe() {
    E || (b(B, b(R, null), !0), b(w, performance.now(), !0), E = !0, we());
  }
  function K() {
    b(B, b(R, null), !0), E && (E = !1);
  }
  Le(() => {
    t.status === "pending" ? fe() : ue(() => {
      K();
    });
  }), Le(() => {
    H && n() && (t.status === "pending" || t.status === "complete") && ml(H, t.autoscroll);
  }), Le(() => {
    o(me) != null && o(B) !== o(me) && (b(M, (performance.now() - o(w)) / 1e3 + o(me)), b(R, o(M).toFixed(1), !0), b(B, o(me), !0));
  });
  function Ge() {
    b(L, !1);
  }
  Le(() => {
    ue(() => {
      Ge();
    }), t.status === "error" && u() && b(L, !0);
  }), Le(() => {
    t.status === "complete" && A() === "output" && P() && d() != null && (b(se, d().toFixed(1), !0), b(W, P() === "full" ? "from cache" : "used cache", !0), b(We, y() != null && y() > d() && y() > 0, !0), b(q, o(We) ? y().toFixed(1) : null, !0), b(C, !0), b(G, !1), Pe && clearTimeout(Pe), Ne && clearTimeout(Ne), Pe = setTimeout(
      () => {
        b(G, !0), Ne = setTimeout(
          () => {
            b(C, !1), b(G, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var Ye = Sl(), Ce = xe(Ye);
  let Ze, Te;
  var ge = ae(Ce);
  {
    var ct = (X) => {
      var re = gl(), ce = ae(re), ne = Z(ce), Ae = ae(ne);
      {
        let Me = ke(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Yn(Ae, {
          get Icon() {
            return Zn;
          },
          get label() {
            return o(Me);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => S(null)
        });
      }
      te(() => ye(ce, `${S() ?? ""} `)), U(X, re);
    };
    $(ge, (X) => {
      S() && v() && X(ct);
    });
  }
  var le = Z(ge, 2);
  {
    var ve = (X) => {
      var re = El(), ce = xe(re);
      {
        var ne = (V) => {
          var Q = vl();
          let He;
          te(() => He = Fe(Q, "", He, {
            transform: `translateX(${(o(St) || 0) * 100 - 100}%)`
          })), U(V, Q);
        };
        $(ce, (V) => {
          c() === "default" && o(Ie) && a() === "full" && V(ne);
        });
      }
      var Ae = Z(ce, 2);
      let Me;
      var Ke = ae(Ae);
      {
        var je = (V) => {
          var Q = Lt(), He = xe(Q);
          Hn(He, 17, l, Sn, (vt, be) => {
            var bt = Lt(), zt = xe(bt);
            {
              var Re = (Ve) => {
                var $e = bl(), et = xe($e);
                {
                  var tt = (De) => {
                    var Oe = qe();
                    te((ze, it) => ye(Oe, `${ze ?? ""}/${it ?? ""}`), [
                      () => Vr(o(be).index || 0),
                      () => Vr(o(be).length)
                    ]), U(De, Oe);
                  }, rt = (De) => {
                    var Oe = qe();
                    te((ze) => ye(Oe, ze), [() => Vr(o(be).index || 0)]), U(De, Oe);
                  };
                  $(et, (De) => {
                    o(be).length != null ? De(tt) : De(rt, -1);
                  });
                }
                var nt = Z(et);
                te(() => ye(nt, ` ${o(be).unit ?? ""} |  `)), U(Ve, $e);
              };
              $(zt, (Ve) => {
                o(be).index != null && Ve(Re);
              });
            }
            U(vt, bt);
          }), U(V, Q);
        }, ht = (V) => {
          var Q = qe();
          te(() => ye(Q, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), U(V, Q);
        }, Je = (V) => {
          var Q = qe("processing |");
          U(V, Q);
        };
        $(Ke, (V) => {
          l() ? V(je) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? V(ht, 1) : t.queue_position === 0 && V(Je, 2);
        });
      }
      var Vt = Z(Ke, 2);
      {
        var dt = (V) => {
          var Q = qe();
          te(() => ye(Q, `${o(oe) ?? ""}${r() ? `/${o(R)}` : ""}s`)), U(V, Q);
        };
        $(Vt, (V) => {
          i() && V(dt);
        });
      }
      var pt = Z(Ae, 2);
      {
        var mt = (V) => {
          var Q = yl(), He = ae(Q), vt = ae(He);
          {
            var be = (Ve) => {
              var $e = Lt(), et = xe($e);
              Hn(et, 17, l, Sn, (tt, rt, nt) => {
                var De = Lt(), Oe = xe(De);
                {
                  var ze = (it) => {
                    var ir = _l(), Xt = xe(ir);
                    {
                      var Pr = (he) => {
                        var Y = qe(" /");
                        U(he, Y);
                      };
                      $(Xt, (he) => {
                        nt !== 0 && he(Pr);
                      });
                    }
                    var Pt = Z(Xt, 2);
                    {
                      var It = (he) => {
                        var Y = qe();
                        te(() => ye(Y, o(rt).desc)), U(he, Y);
                      };
                      $(Pt, (he) => {
                        o(rt).desc != null && he(It);
                      });
                    }
                    var _t = Z(Pt, 2);
                    {
                      var ar = (he) => {
                        var Y = qe("-");
                        U(he, Y);
                      };
                      $(_t, (he) => {
                        o(rt).desc != null && o(ee).progress_level && o(ee).progress_level[nt] != null && he(ar);
                      });
                    }
                    var sr = Z(_t, 2);
                    {
                      var Ir = (he) => {
                        var Y = qe();
                        te((Mr) => ye(Y, `${Mr ?? ""}%`), [
                          () => (100 * (o(ee).progress_level[nt] || 0)).toFixed(1)
                        ]), U(he, Y);
                      };
                      $(sr, (he) => {
                        o(ee).progress_level != null && he(Ir);
                      });
                    }
                    U(it, ir);
                  };
                  $(Oe, (it) => {
                    (o(rt).desc != null || o(ee).progress_level && o(ee).progress_level[nt] != null) && it(ze);
                  });
                }
                U(tt, De);
              }), U(Ve, $e);
            };
            $(vt, (Ve) => {
              l() != null && Ve(be);
            });
          }
          var bt = Z(He, 2), zt = ae(bt);
          let Re;
          te(() => Re = Fe(zt, "", Re, {
            width: `${o(ee).last_progress_level * 100}%`,
            transition: o(ee).progress_bar_transition
          })), U(V, Q);
        }, nr = (V) => {
          {
            let Q = ke(() => c() === "default");
            hl(V, {
              get margin() {
                return o(Q);
              }
            });
          }
        };
        $(pt, (V) => {
          o(ee).last_progress_level != null ? V(mt) : a() === "full" && V(nr, 1);
        });
      }
      var gt = Z(pt, 2);
      {
        var Ht = (V) => {
          var Q = xl(), He = xe(Q), vt = ae(He), be = Z(He, 2);
          Jr(be, t, "additional-loading-text", {}), te(() => ye(vt, f())), U(V, Q);
        };
        $(gt, (V) => {
          i() || V(Ht);
        });
      }
      te(() => Me = ut(Ae, 1, "progress-text svelte-124hqw6", null, Me, {
        "meta-text-center": c() === "center",
        "meta-text": c() === "default"
      })), U(X, re);
    }, Hr = (X) => {
      var re = wl(), ce = xe(re), ne = ae(ce);
      {
        let je = ke(() => t.i18n("common.clear"));
        Yn(ne, {
          get Icon() {
            return Zn;
          },
          get label() {
            return o(je);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var Ae = Z(ce, 2), Me = ae(Ae), Ke = Z(Ae, 2);
      Jr(Ke, t, "error", {}), te((je) => ye(Me, je), [() => t.i18n("common.error")]), U(X, re);
    };
    $(le, (X) => {
      t.status === "pending" ? X(ve) : t.status === "error" && X(Hr, 1);
    });
  }
  mr(Ce, (X) => H = X, () => H);
  var Se = Z(Ce, 2);
  {
    var At = (X) => {
      var re = Tl();
      let ce, ne;
      var Ae = ae(re), Me = Z(Ae);
      {
        var Ke = (ht) => {
          var Je = qe();
          te(() => ye(Je, `~${o(q) ?? ""}s
			→ `)), U(ht, Je);
        };
        $(Me, (ht) => {
          o(We) && ht(Ke);
        });
      }
      var je = Z(Me);
      te(() => {
        ce = ut(re, 1, "cache-indicator svelte-124hqw6", null, ce, { "fade-out": o(G) }), ne = Fe(re, "", ne, { position: p() ? "absolute" : "static" }), ye(Ae, `⚡ ${o(W) ?? ""}: `), ye(je, `${o(se) ?? ""}s`);
      }), U(X, re);
    };
    $(Se, (X) => {
      o(C) && X(At);
    });
  }
  te(() => {
    Ze = ut(Ce, 1, `wrap ${c() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", Ze, {
      "no-click": S() && v(),
      hide: o(ft),
      translucent: c() === "center" && (t.status === "pending" || t.status === "error") || _() || a() === "minimal" || S(),
      generating: t.status === "generating" && a() === "full",
      border: x()
    }), Te = Fe(Ce, "", Te, {
      position: p() ? "absolute" : "static",
      padding: p() ? "0" : "var(--size-8) 0"
    });
  }), U(e, Ye), yr();
}
const Hl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, Pl = [
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
], Il = [
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
Hl([
  Object.fromEntries(Pl.map((e) => [e, ["*"]])),
  Object.fromEntries(Il.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(Ml.map((e) => [e, ["math:*"]]))
]);
er(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var Ol = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), Bl = /* @__PURE__ */ de('<!> <div class="stitch-preview svelte-r41nsf"><div class="canvas-wrap svelte-r41nsf"><canvas tabindex="0" role="application" aria-label="tile stitch preview canvas"></canvas></div> <div class="shortcut-bar svelte-r41nsf">↑↓←→ / WASD 步进 · 右上角圆柄旋转 · 旋转角度数值框微调 · Shift 10x · 普通拖动 1×原图倍率 · Shift+拖精准 0.25× · 背景拖动/Space平移 · +/-缩放 · Esc取消 · Ctrl+Z撤销</div> <div class="status svelte-r41nsf"> </div></div>', 1);
function Nl(e, t) {
  xr(t, !0);
  const r = /* @__PURE__ */ ys(t, Ol), n = 4, i = 70, a = 3, u = 30, l = 1e-3, c = 1, f = 0.25, p = 28, _ = 9, x = 0.01, S = new il(r);
  let v, A, P = z(Qt({ tiles: [], selected: 0 })), d = z(Qt([])), y = z("点击画布以启用键盘"), H = z(!1), E = z("crosshair"), w = z(0), B = z(0), M = z(1), L = z(!1), R = z(!1), C = z(!1), G = z(-1), se = z(!1), q = z(-1), W = 0, We = 0, Pe = 0, Ne = { x: 0, y: 0 }, ft = 0, pe = !1, St = 0, oe = 0, Ie = 0, me = 0, we = 0, ee = 0, fe = !1, K = -1, Ge = 0, Ye = 0, Ce = 0, Ze = !1, Te = "", ge = 0, ct = null, le = [], ve = -1;
  function Hr(s) {
    return typeof s == "number" ? String(s) + "px" : s || "520px";
  }
  function Se(s, h) {
    const g = Number(s);
    return Number.isFinite(g) ? g : h;
  }
  function At(s, h, g) {
    return Math.max(h, Math.min(g, s));
  }
  function X(s) {
    let h = ((s + 180) % 360 + 360) % 360 - 180;
    return h === -180 && (h = 180), Math.abs(h) < x ? 0 : h;
  }
  function re(s) {
    return JSON.parse(JSON.stringify(s || { tiles: [], selected: 0 }));
  }
  function ce(s) {
    var h;
    return {
      index: Math.trunc(Se(s.index, 0)),
      image: (h = s.image) !== null && h !== void 0 ? h : null,
      x: Se(s.x, 0),
      y: Se(s.y, 0),
      width: Math.max(1, Se(s.width, 1)),
      height: Math.max(1, Se(s.height, 1)),
      rotation_deg: X(Se(s.rotation_deg, 0))
    };
  }
  function ne() {
    return Math.trunc(Se(o(P).selected, 0));
  }
  function Ae() {
    const s = Se(o(P).nudge_step, 1);
    return s > 0 ? s : 1;
  }
  function Me() {
    const s = Se(o(P).drag_gain, c);
    return s > 0 ? s : c;
  }
  function Ke() {
    return Math.min(Me(), f);
  }
  function je() {
    return !!o(P).diff_mode;
  }
  function ht() {
    return o(P).show_loupe !== !1;
  }
  function Je() {
    var s, h;
    const g = ne();
    for (const m of o(d))
      if (m.tile.index === g) return m.tile;
    return (h = (s = o(d)[0]) === null || s === void 0 ? void 0 : s.tile) !== null && h !== void 0 ? h : null;
  }
  function Vt() {
    const s = Je();
    if (!s) return { dx: 0, dy: 0 };
    const h = o(d).find((g) => g.tile.index === s.index);
    return h ? {
      dx: Math.round(s.x - h.baseX),
      dy: Math.round(s.y - h.baseY)
    } : { dx: 0, dy: 0 };
  }
  function dt() {
    const s = v?.getBoundingClientRect();
    return {
      width: Math.max(1, s?.width || 1),
      height: Math.max(1, s?.height || 1)
    };
  }
  function pt(s, h) {
    return {
      x: (s - o(w)) / o(M),
      y: (h - o(B)) / o(M)
    };
  }
  function mt(s, h) {
    return {
      x: s * o(M) + o(w),
      y: h * o(M) + o(B)
    };
  }
  function nr(s, h, g) {
    const m = g * Math.PI / 180, T = Math.cos(m), O = Math.sin(m), I = s.x - h.x, D = s.y - h.y;
    return {
      x: h.x + I * T - D * O,
      y: h.y + I * O + D * T
    };
  }
  function gt(s, h = s.x, g = s.y) {
    return { x: h + s.width / 2, y: g + s.height / 2 };
  }
  function Ht(s, h = s.x, g = s.y) {
    const m = gt(s, h, g);
    return [
      { x: h, y: g },
      { x: h + s.width, y: g },
      { x: h + s.width, y: g + s.height },
      { x: h, y: g + s.height }
    ].map((T) => nr(T, m, s.rotation_deg));
  }
  function V(s, h = s.x, g = s.y) {
    const m = Ht(s, h, g);
    return {
      minX: Math.min(...m.map((T) => T.x)),
      minY: Math.min(...m.map((T) => T.y)),
      maxX: Math.max(...m.map((T) => T.x)),
      maxY: Math.max(...m.map((T) => T.y))
    };
  }
  function Q(s, h) {
    const g = nr(h, gt(s), -s.rotation_deg);
    return g.x >= s.x && g.x <= s.x + s.width && g.y >= s.y && g.y <= s.y + s.height;
  }
  function He(s) {
    const h = Ht(s), g = h.reduce((ie, _e) => _e.y < ie.y || Math.abs(_e.y - ie.y) < 1e-6 && _e.x > ie.x ? _e : ie, h[0]), m = gt(s), T = g.x - m.x, O = g.y - m.y, I = Math.max(1, Math.hypot(T, O)), D = p / Math.max(o(M), 0.05);
    return {
      corner: g,
      center: m,
      x: g.x + T / I * D,
      y: g.y + O / I * D,
      radius: _ / Math.max(o(M), 0.05)
    };
  }
  function vt(s, h) {
    const g = Je();
    if (!g) return -1;
    const m = He(g);
    return Math.hypot(s - m.x, h - m.y) <= m.radius ? g.index : -1;
  }
  function be(s) {
    const h = v.getBoundingClientRect();
    return { x: s.clientX - h.left, y: s.clientY - h.top };
  }
  function bt(s, h) {
    return h ? 90 / 255 : s === 0 ? 1 : 140 / 255;
  }
  function zt(s, h, g) {
    if (!s) {
      h === ge && g(null);
      return;
    }
    const m = new Image();
    m.onload = () => {
      h === ge && g(m);
    }, m.onerror = () => {
      h === ge && g(null);
    }, m.src = s;
  }
  function Re() {
    ct && (clearTimeout(ct), ct = null);
  }
  function Ve() {
    return {
      tiles: o(d).map((s) => ({
        index: s.tile.index,
        x: s.tile.x,
        y: s.tile.y,
        rotation_deg: s.tile.rotation_deg
      })),
      selected: ne()
    };
  }
  function $e(s, h) {
    return s.selected === h.selected && s.tiles.length === h.tiles.length && s.tiles.every((g, m) => {
      const T = h.tiles[m];
      return g.index === T.index && Math.abs(g.x - T.x) < 0.01 && Math.abs(g.y - T.y) < 0.01 && Math.abs(g.rotation_deg - T.rotation_deg) < x;
    });
  }
  function et() {
    const s = Ve();
    ve >= 0 && $e(le[ve], s) || (le = le.slice(0, ve + 1), le.push(s), le.length > u && le.shift(), ve = le.length - 1);
  }
  function tt() {
    const s = Ve();
    ve >= 0 && $e(le[ve], s) || (le = le.slice(0, ve + 1), le.push(s), le.length > u && le.shift(), ve = le.length - 1);
  }
  function rt(s) {
    for (const h of s.tiles) {
      const g = o(d).find((m) => m.tile.index === h.index);
      g && (g.tile.x = h.x, g.tile.y = h.y, g.tile.rotation_deg = X(h.rotation_deg));
    }
    b(P, Object.assign(Object.assign({}, o(P)), { selected: s.selected }), !0), b(d, [...o(d)], !0);
  }
  function nt() {
    if (!o(d).length) {
      b(w, 20), b(B, 20), b(M, 1);
      return;
    }
    let s = 1 / 0, h = 1 / 0, g = -1 / 0, m = -1 / 0;
    for (const Or of o(d)) {
      const Xe = V(Or.tile);
      s = Math.min(s, Xe.minX), h = Math.min(h, Xe.minY), g = Math.max(g, Xe.maxX), m = Math.max(m, Xe.maxY);
    }
    const T = 40, O = Math.max(1, g - s), I = Math.max(1, m - h), { width: D, height: ie } = dt(), _e = At(Math.min((D - T * 2) / O, (ie - T * 2) / I), 0.05, 8);
    b(M, _e, !0), b(w, (D - (s + g) * _e) / 2), b(B, (ie - (h + m) * _e) / 2);
  }
  function De(s) {
    Re(), ge += 1;
    const h = ge, g = new Map(o(d).map((I) => [I.tile.index, I])), m = re(s), T = Array.isArray(m.tiles) ? m.tiles.map((I) => {
      const D = ce(I);
      if (!D.image) {
        const ie = g.get(D.index);
        ie?.tile.image && (D.image = ie.tile.image);
      }
      return D;
    }) : [];
    b(
      P,
      Object.assign(Object.assign({}, m), {
        tiles: T,
        selected: T.length ? Math.trunc(Se(m.selected, T[0].index)) : 0,
        nudge_step: Se(m.nudge_step, 1),
        diff_mode: !!m.diff_mode,
        show_loupe: m.show_loupe !== !1,
        drag_gain: Se(m.drag_gain, c),
        status: m.status || ""
      }),
      !0
    ), b(C, !1), b(G, -1), b(se, !1), b(q, -1), b(R, !1), le = [], ve = -1;
    const O = T.map((I) => ({
      tile: Object.assign({}, I),
      image: null,
      ready: !1,
      baseX: I.x,
      baseY: I.y
    }));
    b(d, O, !0), O.length && tt(), b(y, o(P).status || (T.length ? "点击画布以启用键盘" : "等待 tile 数据"), !0);
    for (let I = 0; I < O.length; I++) {
      const D = O[I];
      zt(D.tile.image, h, (ie) => {
        if (h !== ge) return;
        const _e = o(d)[I];
        !_e || _e.tile.index !== D.tile.index || (_e.image = ie, _e.ready = !!ie, b(d, [...o(d)], !0), Y());
      });
    }
    requestAnimationFrame(() => nt());
  }
  Le(() => {
    const s = JSON.stringify(S.props.value || null);
    s !== Te && (Te = s, De(S.props.value));
  }), Qa(() => {
    ge += 1, Re();
  }), ci(() => (window.addEventListener("blur", ar), () => window.removeEventListener("blur", ar)));
  function Oe(s) {
    var h;
    const g = Object.assign(Object.assign({}, o(P)), {
      tiles: o(d).map((m) => Object.assign({}, m.tile)),
      selected: ne(),
      status: (h = s ?? o(P).status) !== null && h !== void 0 ? h : ""
    });
    S.props.value = g, Te = JSON.stringify(g);
  }
  function ze(s, h = !0) {
    b(P, Object.assign(Object.assign({}, o(P)), { status: s }), !0), b(y, s, !0), Oe(s), h && (Re(), S.dispatch("change")), Y();
  }
  function it(s, h = 140) {
    ze(s, !1), Re(), ct = setTimeout(
      () => {
        ct = null, S.dispatch("change");
      },
      h
    );
  }
  function ir(s, h) {
    for (let g = o(d).length - 1; g >= 0; g--) {
      const m = o(d)[g].tile;
      if (Q(m, { x: s, y: h }))
        return m.index;
    }
    return -1;
  }
  function Xt(s, h) {
    s < 0 || (b(P, Object.assign(Object.assign({}, o(P)), { selected: s }), !0), b(y, h || "已选择 tile " + String(s), !0), Y());
  }
  function Pr(s, h) {
    const g = Je();
    if (!g) return;
    et(), g.x += s, g.y += h, b(d, [...o(d)], !0), tt();
    const m = Vt();
    it("微调 tile " + String(g.index) + " → dx=" + String(m.dx) + " dy=" + String(m.dy));
  }
  function Pt(s, h, g) {
    const m = pt(s, h);
    b(M, At(o(M) * g, 0.05, 16), !0);
    const T = mt(m.x, m.y);
    b(w, o(w) + (s - T.x)), b(B, o(B) + (h - T.y)), Y();
  }
  function It(s) {
    if (!(!v || s < 0))
      try {
        v.hasPointerCapture(s) && v.releasePointerCapture(s);
      } catch {
      }
  }
  function _t(s = "已取消拖动") {
    const h = o(R) || K >= 0 || o(G) >= 0 || o(q) >= 0, g = K, m = o(d).find((O) => O.tile.index === o(G));
    m && (o(C) || fe) && (m.tile.x = W, m.tile.y = We, b(d, [...o(d)], !0));
    const T = o(d).find((O) => O.tile.index === o(q));
    T && (o(se) || pe) && (T.tile.rotation_deg = Pe, b(d, [...o(d)], !0)), h && Re(), b(P, Object.assign(Object.assign({}, o(P)), { selected: Ge }), !0), b(C, !1), b(G, -1), b(se, !1), b(q, -1), fe = !1, pe = !1, b(R, !1), K = -1, It(g), h && (b(y, s, !0), Oe(s), Y());
  }
  function ar() {
    _t("窗口失焦，已取消拖动");
  }
  function sr(s, h, g = h.tile.x, m = h.tile.y) {
    if (!h.image) return;
    const T = h.tile, O = gt(T, g, m);
    s.save(), s.translate(O.x, O.y), s.rotate(T.rotation_deg * Math.PI / 180), s.drawImage(h.image, -T.width / 2, -T.height / 2, T.width, T.height), s.restore();
  }
  function Ir(s) {
    if (!ht() || !Ze) return;
    const { width: h, height: g } = dt(), m = At(Ye, i + 2, h - i - 2), T = At(Ce, i + 2, g - i - 2), O = pt(m, T);
    s.save(), s.beginPath(), s.arc(m, T, i, 0, Math.PI * 2), s.clip(), s.fillStyle = "#0f172a", s.fillRect(m - i, T - i, i * 2, i * 2), s.translate(m, T), s.scale(a * o(M), a * o(M)), s.translate(-O.x, -O.y);
    for (const I of o(d)) {
      if (!I.ready || !I.image) continue;
      const D = I.tile, ie = o(C) && o(G) === D.index;
      s.globalAlpha = bt(D.index, ie), je() && D.index !== 0 ? s.globalCompositeOperation = "difference" : s.globalCompositeOperation = "source-over", sr(s, I);
    }
    s.restore(), s.save(), s.beginPath(), s.arc(m, T, i, 0, Math.PI * 2), s.strokeStyle = "rgba(255,255,255,0.9)", s.lineWidth = 2, s.stroke(), s.strokeStyle = "rgba(15,23,42,0.85)", s.lineWidth = 1, s.beginPath(), s.moveTo(m - 8, T), s.lineTo(m + 8, T), s.moveTo(m, T - 8), s.lineTo(m, T + 8), s.stroke(), s.restore();
  }
  function he(s) {
    const h = Je();
    if (s.save(), h) {
      const g = Ht(h).map((I) => mt(I.x, I.y));
      s.strokeStyle = "#22d3ee", s.lineWidth = 2, s.beginPath(), s.moveTo(g[0].x, g[0].y);
      for (let I = 1; I < g.length; I++) s.lineTo(g[I].x, g[I].y);
      s.closePath(), s.stroke();
      const m = He(h), T = mt(m.corner.x, m.corner.y), O = mt(m.x, m.y);
      s.strokeStyle = "rgba(226,232,240,0.9)", s.lineWidth = 1.5, s.beginPath(), s.moveTo(T.x, T.y), s.lineTo(O.x, O.y), s.stroke(), s.beginPath(), s.arc(O.x, O.y, _, 0, Math.PI * 2), s.fillStyle = o(se) ? "#f59e0b" : "#f8fafc", s.fill(), s.strokeStyle = "#22d3ee", s.lineWidth = 2, s.stroke();
    }
    s.restore();
  }
  function Y() {
    if (!v) return;
    const s = window.devicePixelRatio || 1, { width: h, height: g } = dt();
    v.width = Math.max(1, Math.round(h * s)), v.height = Math.max(1, Math.round(g * s));
    const m = v.getContext("2d");
    if (m) {
      if (m.setTransform(s, 0, 0, s, 0, 0), m.clearRect(0, 0, h, g), m.fillStyle = "#0f172a", m.fillRect(0, 0, h, g), !o(d).length) {
        m.fillStyle = "#94a3b8", m.font = "16px sans-serif", m.fillText("等待 tile 数据", 24, 40);
        return;
      }
      m.save(), m.translate(o(w), o(B)), m.scale(o(M), o(M));
      for (const T of o(d)) {
        if (!T.ready || !T.image) continue;
        const O = T.tile, I = o(C) && o(G) === O.index;
        m.globalAlpha = bt(O.index, I), je() && O.index !== 0 ? m.globalCompositeOperation = "difference" : m.globalCompositeOperation = "source-over", sr(m, T);
      }
      if (m.restore(), o(C) && o(G) >= 0) {
        const T = o(d).find((O) => O.tile.index === o(G));
        if (T) {
          const O = Ht(T.tile, St, oe).map((I) => mt(I.x, I.y));
          m.save(), m.setLineDash([6, 4]), m.strokeStyle = "rgba(250,204,21,0.95)", m.lineWidth = 2, m.beginPath(), m.moveTo(O[0].x, O[0].y);
          for (let I = 1; I < O.length; I++) m.lineTo(O[I].x, O[I].y);
          m.closePath(), m.stroke(), m.restore();
        }
      }
      he(m), Ir(m);
    }
  }
  function Mr() {
    b(H, !0), b(y, "键盘已接管");
  }
  function Gi() {
    _t(), b(H, !1), b(L, !1), !o(C) && !o(R) && b(E, "crosshair"), b(y, o(P).status || "点击画布以启用键盘", !0);
  }
  function ji() {
    v.focus();
  }
  function Vi(s) {
    if (!o(H)) return;
    const h = s.key, g = h.toLowerCase(), m = /* @__PURE__ */ new Set([
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "w",
      "a",
      "s",
      "d",
      " ",
      "+",
      "=",
      "-",
      "_",
      "Escape",
      "z"
    ]);
    if ((m.has(h) || m.has(g) || s.ctrlKey && g === "z") && s.preventDefault(), h === " " || h === "Spacebar") {
      b(L, !0), b(E, "grab");
      return;
    }
    if (h === "Escape") {
      (o(C) || o(se) || o(R) || K >= 0) && _t("已取消拖动");
      return;
    }
    if (s.ctrlKey && g === "z") {
      ve > 0 && (ve -= 1, rt(le[ve]), it("撤销到步骤 " + String(ve + 1)));
      return;
    }
    if (h === "+" || h === "=") {
      const I = dt();
      Pt(I.width / 2, I.height / 2, 1.15);
      return;
    }
    if (h === "-" || h === "_") {
      const I = dt();
      Pt(I.width / 2, I.height / 2, 1 / 1.15);
      return;
    }
    let T = 0, O = 0;
    if ((h === "ArrowLeft" || g === "a") && (T = -1), (h === "ArrowRight" || g === "d") && (T = 1), (h === "ArrowUp" || g === "w") && (O = -1), (h === "ArrowDown" || g === "s") && (O = 1), T !== 0 || O !== 0) {
      const I = Ae() * (s.shiftKey ? 10 : 1);
      Pr(T * I, O * I);
    }
  }
  function zi(s) {
    o(H) && (s.key === " " || s.key === "Spacebar") && (b(L, !1), o(R) || b(E, o(C) ? "grabbing" : "crosshair", !0));
  }
  function Xi(s) {
    if (!v) return;
    Re();
    const h = be(s);
    if (Ie = h.x, me = h.y, we = h.x, ee = h.y, Ye = h.x, Ce = h.y, Ze = !0, s.shiftKey, Ge = ne(), s.button === 1 || s.button === 0 && o(L)) {
      b(R, !0), K = s.pointerId, b(E, "grabbing"), v.setPointerCapture(s.pointerId);
      return;
    }
    if (s.button !== 0) return;
    const g = pt(h.x, h.y), m = vt(g.x, g.y);
    if (m >= 0) {
      const O = o(d).find((I) => I.tile.index === m);
      if (O) {
        const I = O.tile;
        b(q, m, !0), Pe = I.rotation_deg, Ne = gt(I), ft = Math.atan2(g.y - Ne.y, g.x - Ne.x) * 180 / Math.PI, b(se, !0), pe = !1, K = s.pointerId, b(E, "grabbing"), v.setPointerCapture(s.pointerId), Y();
        return;
      }
    }
    const T = ir(g.x, g.y);
    if (T >= 0) {
      Xt(T);
      const O = o(d).find((I) => I.tile.index === T);
      O && (b(G, T, !0), W = O.tile.x, We = O.tile.y, St = O.tile.x, oe = O.tile.y, b(C, !1), fe = !1, K = s.pointerId, v.setPointerCapture(s.pointerId));
    } else
      b(G, -1), b(R, !0), K = s.pointerId, b(E, "grabbing"), v.setPointerCapture(s.pointerId);
    Y();
  }
  function qi(s) {
    const h = be(s);
    if (Ye = h.x, Ce = h.y, Ze = !0, o(R)) {
      b(w, o(w) + (h.x - we)), b(B, o(B) + (h.y - ee)), we = h.x, ee = h.y, b(E, "grabbing"), Y();
      return;
    }
    if (K === s.pointerId && o(q) >= 0) {
      const g = o(d).find((O) => O.tile.index === o(q));
      if (!g) return;
      const m = pt(h.x, h.y), T = Math.hypot(h.x - Ie, h.y - me);
      if (!pe && T >= n && (pe = !0, et(), b(E, "grabbing")), pe) {
        const O = Math.atan2(m.y - Ne.y, m.x - Ne.x) * 180 / Math.PI;
        g.tile.rotation_deg = X(Pe + O - ft), b(d, [...o(d)], !0), b(y, "旋转 tile " + String(o(q)) + "  " + g.tile.rotation_deg.toFixed(1) + "°"), b(P, Object.assign(Object.assign({}, o(P)), { status: o(y) }), !0), Oe(o(y));
      }
      we = h.x, ee = h.y, Y();
      return;
    }
    if (K === s.pointerId && o(G) >= 0) {
      const g = Math.hypot(h.x - Ie, h.y - me), m = o(d).find((T) => T.tile.index === o(G));
      if (!m) return;
      if (!o(C) && g >= n && (b(C, !0), et(), b(E, "grabbing")), o(C)) {
        s.shiftKey;
        const T = s.shiftKey ? Ke() : Me(), O = (h.x - we) / o(M) * T, I = (h.y - ee) / o(M) * T;
        m.tile.x += O, m.tile.y += I, b(d, [...o(d)], !0), fe = !0;
        const D = Vt();
        b(y, "拖动 tile " + String(o(G)) + "  dx=" + String(D.dx) + " dy=" + String(D.dy)), b(P, Object.assign(Object.assign({}, o(P)), { status: o(y) }), !0), Oe(o(y));
      }
      we = h.x, ee = h.y, Y();
      return;
    }
    o(L) ? b(E, "grab") : b(E, "crosshair"), Y();
  }
  function Wi(s) {
    var h;
    if (o(R)) {
      b(R, !1), It(s.pointerId), b(E, o(L) ? "grab" : "crosshair", !0), K = -1, Y();
      return;
    }
    if (K === s.pointerId && o(q) >= 0) {
      const g = o(q);
      if (pe) {
        tt();
        const m = (h = o(d).find((T) => T.tile.index === g)) === null || h === void 0 ? void 0 : h.tile;
        ze("tile " + String(g) + " 旋转=" + (m ? m.rotation_deg.toFixed(1) : "0.0") + "°");
      }
      b(se, !1), b(q, -1), pe = !1, K = -1, It(s.pointerId), b(E, o(L) ? "grab" : "crosshair", !0), Y();
      return;
    }
    if (K === s.pointerId && o(G) >= 0) {
      const g = be(s), m = Math.hypot(g.x - Ie, g.y - me);
      if (!o(C) && m < n)
        Xt(o(G), "已选择 tile " + String(o(G))), ze("已选择 tile " + String(o(G)));
      else if (o(C) && fe) {
        tt();
        const T = Vt();
        ze("tile " + String(o(G)) + " 对齐 dx=" + String(T.dx) + " dy=" + String(T.dy));
      }
      b(C, !1), b(G, -1), fe = !1, K = -1, It(s.pointerId), b(E, o(L) ? "grab" : "crosshair", !0), Y();
    }
  }
  function Yi() {
    _t();
  }
  function Zi() {
    Ze = !1, !o(R) && !o(C) && !o(se) && b(E, o(L) ? "grab" : "crosshair", !0), Y();
  }
  function Ji(s) {
    s.preventDefault();
    const h = be(s), g = Math.exp(-s.deltaY * (s.ctrlKey ? l * 4 : l));
    Pt(h.x, h.y, g);
  }
  {
    let s = ke(() => o(H) ? "focus" : "base");
    ol(e, {
      get visible() {
        return S.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return o(s);
      },
      padding: !1,
      get elem_id() {
        return S.shared.elem_id;
      },
      get elem_classes() {
        return S.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return S.shared.container;
      },
      get scale() {
        return S.shared.scale;
      },
      get min_width() {
        return S.shared.min_width;
      },
      children: (h, g) => {
        var m = Bl(), T = xe(m);
        Al(T, Es(
          {
            get autoscroll() {
              return S.shared.autoscroll;
            },
            get i18n() {
              return S.i18n;
            }
          },
          () => S.shared.loading_status,
          {
            on_clear_status: () => S.dispatch("clear_status", S.shared.loading_status)
          }
        ));
        var O = Z(T, 2), I = ae(O), D = ae(I);
        let ie;
        mr(D, (Xe) => v = Xe, () => v), mr(I, (Xe) => A = Xe, () => A);
        var _e = Z(I, 4), Or = ae(_e);
        te(
          (Xe) => {
            Fe(O, Xe), Fe(D, "cursor:" + o(E)), ie = ut(D, 1, "svelte-r41nsf", null, ie, { focused: o(H) }), ye(Or, o(y));
          },
          [() => "height:" + Hr(S.props.height)]
        ), Ot("focus", D, Mr), Ot("blur", D, Gi), st("click", D, ji), st("keydown", D, Vi), st("keyup", D, zi), st("pointerdown", D, Xi), st("pointermove", D, qi), st("pointerup", D, Wi), Ot("pointercancel", D, Yi), Ot("pointerleave", D, Zi), Ot("wheel", D, Ji), U(h, m);
      },
      $$slots: { default: !0 }
    });
  }
  yr();
}
er([
  "click",
  "keydown",
  "keyup",
  "pointerdown",
  "pointermove",
  "pointerup"
]);
export {
  Nl as default
};
