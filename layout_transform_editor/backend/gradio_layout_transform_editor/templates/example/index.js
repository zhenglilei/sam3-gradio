import { e as x, c as k, a as E, b as D, g as Y, i as j, T as q, r as M, d as b, p as G, f as w, h as z, m as J, s as K, j as Q, E as V, C as P, A as W, N as X, k as Z, l as $, n as ee, u as te, o as L, q as y, t as I, v as re, w as h, x as H, y as se, z as ne, B as ae, P as ie, D as fe, F as oe, G as ce, H as le, I as ue, J as de, K as _e, L as he, S as ve, M as ge, O as me, Q as pe, R as be, U as Te, V as Ee } from "./runtime-CpL30OSY.js";
x();
let m = !1;
function Se(t) {
  var e = m;
  try {
    return m = !1, [t(), m];
  } finally {
    m = e;
  }
}
const Ae = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (t) => t
  })
);
function Me(t) {
  return (
    /** @type {string} */
    Ae?.createHTML(t) ?? t
  );
}
function we(t) {
  var e = k("template");
  return e.innerHTML = Me(t.replaceAll("<!>", "<!---->")), e.content;
}
function U(t, e) {
  var r = (
    /** @type {Effect} */
    D
  );
  r.nodes === null && (r.nodes = { start: t, end: e, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function Pe(t, e) {
  var r = (e & q) !== 0, s, n = !t.startsWith("<!>");
  return () => {
    s === void 0 && (s = we(n ? t : "<!>" + t), s = /** @type {TemplateNode} */
    Y(s));
    var i = (
      /** @type {TemplateNode} */
      r || j ? document.importNode(s, !0) : s.cloneNode(!0)
    );
    return U(i, i), i;
  };
}
function Le() {
  var t = document.createDocumentFragment(), e = document.createComment(""), r = E();
  return t.append(e, r), U(e, r), t;
}
function C(t, e) {
  t !== null && t.before(
    /** @type {Node} */
    e
  );
}
class ye {
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
  #s = /* @__PURE__ */ new Set();
  /**
   * Whether to pause (i.e. outro) on change, or destroy immediately.
   * This is necessary for `<svelte:element>`
   */
  #n = !0;
  /**
   * @param {TemplateNode} anchor
   * @param {boolean} transition
   */
  constructor(e, r = !0) {
    this.anchor = e, this.#n = r;
  }
  /**
   * @param {Batch} batch
   */
  #a = (e) => {
    if (this.#t.has(e)) {
      var r = (
        /** @type {Key} */
        this.#t.get(e)
      ), s = this.#r.get(r);
      if (s)
        M(s), this.#s.delete(r);
      else {
        var n = this.#e.get(r);
        n && (M(n.effect), this.#r.set(r, n.effect), this.#e.delete(r), n.fragment.lastChild.remove(), this.anchor.before(n.fragment), s = n.effect);
      }
      for (const [i, a] of this.#t) {
        if (this.#t.delete(i), i === e)
          break;
        const f = this.#e.get(a);
        f && (b(f.effect), this.#e.delete(a));
      }
      for (const [i, a] of this.#r) {
        if (i === r || this.#s.has(i)) continue;
        const f = () => {
          if (Array.from(this.#t.values()).includes(i)) {
            var c = document.createDocumentFragment();
            J(a, c), c.append(E()), this.#e.set(i, { effect: a, fragment: c });
          } else
            b(a);
          this.#s.delete(i), this.#r.delete(i);
        };
        this.#n || !s ? (this.#s.add(i), G(a, f, !1)) : f();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #i = (e) => {
    this.#t.delete(e);
    const r = Array.from(this.#t.values());
    for (const [s, n] of this.#e)
      r.includes(s) || (b(n.effect), this.#e.delete(s));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(e, r) {
    var s = (
      /** @type {Batch} */
      z
    ), n = K();
    if (r && !this.#r.has(e) && !this.#e.has(e))
      if (n) {
        var i = document.createDocumentFragment(), a = E();
        i.append(a), this.#e.set(e, {
          effect: w(() => r(a)),
          fragment: i
        });
      } else
        this.#r.set(
          e,
          w(() => r(this.anchor))
        );
    if (this.#t.set(s, e), n) {
      for (const [f, o] of this.#r)
        f === e ? s.unskip_effect(o) : s.skip_effect(o);
      for (const [f, o] of this.#e)
        f === e ? s.unskip_effect(o.effect) : s.skip_effect(o.effect);
      s.oncommit(this.#a), s.ondiscard(this.#i);
    } else
      this.#a(s);
  }
}
function Ce(t, e, r = !1) {
  var s = new ye(t), n = r ? V : 0;
  function i(a, f) {
    s.ensure(a, f);
  }
  Q(() => {
    var a = !1;
    e((f, o = 0) => {
      a = !0, i(o, f);
    }), a || i(-1, null);
  }, n);
}
const N = [...` 	
\r\f \v\uFEFF`];
function Ne(t, e, r) {
  var s = "" + t;
  if (r) {
    for (var n of Object.keys(r))
      if (r[n])
        s = s ? s + " " + n : n;
      else if (s.length)
        for (var i = n.length, a = 0; (a = s.indexOf(n, a)) >= 0; ) {
          var f = a + i;
          (a === 0 || N.includes(s[a - 1])) && (f === s.length || N.includes(s[f])) ? s = (a === 0 ? "" : s.substring(0, a)) + s.substring(f + 1) : a = f;
        }
  }
  return s === "" ? null : s;
}
function Oe(t, e, r, s, n, i) {
  var a = (
    /** @type {any} */
    t[P]
  );
  if (a !== r || a === void 0) {
    var f = Ne(r, s, i);
    f == null ? t.removeAttribute("class") : t.className = f, t[P] = r;
  } else if (i && n !== i)
    for (var o in i) {
      var c = !!i[o];
      (n == null || c !== !!n[o]) && t.classList.toggle(o, c);
    }
  return i;
}
const Re = /* @__PURE__ */ Symbol("is custom element"), De = /* @__PURE__ */ Symbol("is html");
function Ie(t, e, r, s) {
  var n = He(t);
  n[e] !== (n[e] = r) && (r == null ? t.removeAttribute(e) : typeof r != "string" && Ue(t).includes(e) ? t[e] = r : t.setAttribute(e, r));
}
function He(t) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    t[W] ??= {
      [Re]: t.nodeName.includes("-"),
      [De]: t.namespaceURI === X
    }
  );
}
var O = /* @__PURE__ */ new Map();
function Ue(t) {
  var e = t.getAttribute("is") || t.nodeName, r = O.get(e);
  if (r) return r;
  O.set(e, r = []);
  for (var s, n = t, i = Element.prototype; i !== n; ) {
    s = $(n);
    for (var a in s)
      s[a].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      a !== "innerHTML" && a !== "textContent" && a !== "innerText" && r.push(a);
    n = Z(n);
  }
  return r;
}
function Be(t = !1) {
  const e = (
    /** @type {ComponentContextLegacy} */
    ee
  ), r = e.l.u;
  if (!r) return;
  let s = () => H(e.s);
  if (t) {
    let n = 0, i = (
      /** @type {Record<string, any>} */
      {}
    );
    const a = se(() => {
      let f = !1;
      const o = e.s;
      for (const c in o)
        o[c] !== i[c] && (i[c] = o[c], f = !0);
      return f && n++, n;
    });
    s = () => h(a);
  }
  r.b.length && te(() => {
    R(e, s), y(r.b);
  }), L(() => {
    const n = I(() => r.m.map(re));
    return () => {
      for (const i of n)
        typeof i == "function" && i();
    };
  }), r.a.length && L(() => {
    R(e, s), y(r.a);
  });
}
function R(t, e) {
  if (t.l.s)
    for (const r of t.l.s) h(r);
  e();
}
function T(t, e, r, s) {
  var n = !le || (r & ue) !== 0, i = (r & de) !== 0, a = (
    /** @type {V} */
    s
  ), f = !0, o = () => (f && (f = !1, a = /** @type {V} */
  s), a);
  let c;
  {
    var v = ve in t || ge in t;
    c = ne(t, e)?.set ?? (v && e in t ? (l) => t[e] = l : void 0);
  }
  var u, S = !1;
  [u, S] = Se(() => (
    /** @type {V} */
    t[e]
  )), u === void 0 && s !== void 0 && (u = o(), c && (n && ae(), c(u)));
  var d;
  if (n ? d = () => {
    var l = (
      /** @type {V} */
      t[e]
    );
    return l === void 0 ? o() : (f = !0, l);
  } : d = () => {
    var l = (
      /** @type {V} */
      t[e]
    );
    return l !== void 0 && (a = /** @type {V} */
    void 0), l === void 0 ? a : l;
  }, n && (r & ie) === 0)
    return d;
  if (c) {
    var B = t.$$legacy;
    return (
      /** @type {() => V} */
      (function(l, g) {
        return arguments.length > 0 ? ((!n || !g || B || S) && c(g ? d() : l), l) : d();
      })
    );
  }
  var p = !1, _ = _e(() => (p = !1, d()));
  h(_);
  var F = (
    /** @type {Effect} */
    D
  );
  return (
    /** @type {() => V} */
    (function(l, g) {
      if (arguments.length > 0) {
        const A = g ? h(_) : n && i ? fe(l) : l;
        return oe(_, A), p = !0, a !== void 0 && (a = A), l;
      }
      return he && p || (F.f & ce) !== 0 ? _.v : h(_);
    })
  );
}
var Fe = /* @__PURE__ */ Pe('<div><img alt="" class="svelte-s3apn9"/></div>');
function ke(t, e) {
  be(e, !1);
  let r = T(e, "value", 8), s = T(e, "type", 8), n = T(e, "selected", 8, !1);
  Be();
  var i = Le(), a = me(i);
  {
    var f = (o) => {
      var c = Fe();
      let v;
      var u = Ee(c);
      Te(() => {
        v = Oe(c, 1, "container svelte-s3apn9", null, v, {
          table: s() === "table",
          gallery: s() === "gallery",
          selected: n()
        }), Ie(u, "src", (H(r()), I(() => r().url)));
      }), C(o, c);
    };
    Ce(a, (o) => {
      r() && o(f);
    });
  }
  C(t, i), pe();
}
export {
  ke as default
};
