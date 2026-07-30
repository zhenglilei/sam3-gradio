import { e as O, c as y, g as L, i as D, T as R, a as S, b as x, P as A, d as g, p as M, s as N, D as w, l as I, f as B, h as H, j as U, k as Y, S as $, L as j, t as q, m as C, u as G, n as J, o as W } from "./render-CYmqD-1W.js";
O();
let d = !1;
function z(e) {
  var r = d;
  try {
    return d = !1, [e(), d];
  } finally {
    d = r;
  }
}
const F = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function K(e) {
  return (
    /** @type {string} */
    F?.createHTML(e) ?? e
  );
}
function Q(e) {
  var r = y("template");
  return r.innerHTML = K(e.replaceAll("<!>", "<!---->")), r.content;
}
function V(e, r) {
  var n = (
    /** @type {Effect} */
    S
  );
  n.nodes === null && (n.nodes = { start: e, end: r, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function X(e, r) {
  var n = (r & R) !== 0, a, i = !e.startsWith("<!>");
  return () => {
    a === void 0 && (a = Q(i ? e : "<!>" + e), a = /** @type {TemplateNode} */
    L(a));
    var s = (
      /** @type {TemplateNode} */
      n || D ? document.importNode(a, !0) : a.cloneNode(!0)
    );
    return V(s, s), s;
  };
}
function Z(e, r) {
  e !== null && e.before(
    /** @type {Node} */
    r
  );
}
function k(e, r, n, a) {
  var i = !I || (n & B) !== 0, s = (n & H) !== 0, u = (
    /** @type {V} */
    a
  ), o = !0, b = () => (o && (o = !1, u = /** @type {V} */
  a), u);
  let c;
  {
    var P = $ in e || j in e;
    c = x(e, r)?.set ?? (P && r in e ? (t) => e[r] = t : void 0);
  }
  var E, m = !1;
  [E, m] = z(() => (
    /** @type {V} */
    e[r]
  ));
  var f;
  if (i ? f = () => {
    var t = (
      /** @type {V} */
      e[r]
    );
    return t === void 0 ? b() : (o = !0, t);
  } : f = () => {
    var t = (
      /** @type {V} */
      e[r]
    );
    return t !== void 0 && (u = /** @type {V} */
    void 0), t === void 0 ? u : t;
  }, i && (n & A) === 0)
    return f;
  if (c) {
    var h = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(t, _) {
        return arguments.length > 0 ? ((!i || !_ || h || m) && c(_ ? f() : t), t) : f();
      })
    );
  }
  var v = !1, l = U(() => (v = !1, f()));
  g(l);
  var p = (
    /** @type {Effect} */
    S
  );
  return (
    /** @type {() => V} */
    (function(t, _) {
      if (arguments.length > 0) {
        const T = _ ? g(l) : i && s ? M(t) : t;
        return N(l, T), v = !0, u !== void 0 && (u = T), t;
      }
      return Y && v || (p.f & w) !== 0 ? l.v : g(l);
    })
  );
}
var ee = /* @__PURE__ */ X("<pre> </pre>");
function te(e, r) {
  let n = k(r, "value", 8);
  var a = ee(), i = W(a);
  q((s) => J(i, s), [
    () => (C(n()), G(() => JSON.stringify(n(), null, 2)))
  ]), Z(e, a);
}
export {
  te as default
};
