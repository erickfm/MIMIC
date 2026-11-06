import{P as ci,x as li,a as hi,t as mi,s as pi,b as E,p as di,c as bi,V as K,d as _t,e as lt,f as xi,l as wi,T as F,g as At,h as gi,X as q,z as ut,i as Pt,G as ft,C as vi,j as yi,k as _i,m as kt,n as Ai,o as ki}from"./gx.a6e4194c.js";import"./modulepreload-polyfill.c7c6310f.js";import"./moverlay.8ef32501.js";const Ft=`
// run-time loop bounds (the program sets uK to (4, 25, 3, 2)): a loop the compiler cannot unroll keeps the shader small
uniform ivec4 uK;
const uint F_NAN = 0x7fc00000u;
const uint F_ONE = 0x3f800000u;
const uint F_INF = 0x7f800000u;

// index of the highest set bit of m (m != 0); GLSL ES 3.00 has no findMSB
int f_msb(uint m) {
  int r = 0;
  bool c16 = m >= 0x10000u; r += c16 ? 16 : 0; m = c16 ? m >> 16 : m;
  bool c8 = m >= 0x100u; r += c8 ? 8 : 0; m = c8 ? m >> 8 : m;
  bool c4 = m >= 0x10u; r += c4 ? 4 : 0; m = c4 ? m >> 4 : m;
  bool c2 = m >= 4u; r += c2 ? 2 : 0; m = c2 ? m >> 2 : m;
  r += m >= 2u ? 1 : 0;
  return r;
}

// pack a rounded result: sign, biased exponent e, 24-bit mantissa m (bit 23 set), guard bit and sticky (0 or 1)
uint f_pack(uint s, int e, uint m, uint guard, uint sticky) {
  uint up = guard & (sticky | (m & 1u));
  m += up;
  bool carry = m == 0x1000000u;
  m = carry ? 0x800000u : m;
  e += carry ? 1 : 0;
  uint r = (uint(clamp(e, 0, 255)) << 23) | (m & 0x7fffffu);
  r = e >= 255 ? F_INF : r;
  r = e <= 0 ? 0u : r;
  return (s << 31) | r;
}

uint f_neg(uint a) { return a ^ 0x80000000u; }
uint f_abs(uint a) { return a & 0x7fffffffu; }

uint f_add(uint a, uint b) {
  uint aa = a & 0x7fffffffu, ab = b & 0x7fffffffu;
  bool sw = aa < ab;
  uint x = sw ? b : a, y = sw ? a : b;
  uint ax = sw ? ab : aa, ay = sw ? aa : ab;
  uint sx = x >> 31, sy = y >> 31;
  int ex = int(ax >> 23), ey = int(ay >> 23);
  uint mx = ex == 0 ? 0u : (((ax & 0x7fffffu) | 0x800000u) << 3);
  uint my = ey == 0 ? 0u : (((ay & 0x7fffffu) | 0x800000u) << 3);
  uint sh = uint(min(ex - ey, 31));
  uint lost = my & ((1u << sh) - 1u);
  my = (my >> sh) | (lost != 0u ? 1u : 0u);
  uint m = sx == sy ? mx + my : mx - my;
  bool ov = m >= 0x8000000u;
  int lz = (ov || m == 0u) ? 0 : max(0, 26 - f_msb(m));
  uint mn = ov ? ((m >> 1) | (m & 1u)) : (m << uint(lz));
  int e = ex + (ov ? 1 : 0) - lz;
  uint r = f_pack(sx, e, mn >> 3, (mn >> 2) & 1u, (mn & 3u) != 0u ? 1u : 0u);
  r = m == 0u ? 0u : r;
  r = ex == 0 ? (a & b & 0x80000000u) : r;
  // infinities and not-a-numbers
  uint sp = (ax > F_INF || (ay == F_INF && sx != sy)) ? F_NAN : x;
  return ex == 255 ? sp : r;
}

uint f_sub(uint a, uint b) { return f_add(a, f_neg(b)); }

uint f_mul(uint a, uint b) {
  uint s = (a ^ b) >> 31;
  uint aa = a & 0x7fffffffu, ab = b & 0x7fffffffu;
  int ea = int(aa >> 23), eb = int(ab >> 23);
  int e = ea + eb - 127;
  uint ma = (aa & 0x7fffffu) | 0x800000u, mb = (ab & 0x7fffffu) | 0x800000u;
  uint ah = ma >> 12, al = ma & 0xfffu, bh = mb >> 12, bl = mb & 0xfffu;
  uint ll = al * bl, mid = al * bh + ah * bl, hh = ah * bh;
  uint lo = ll + ((mid & 0xfffu) << 12);
  uint hi = hh + (mid >> 12) + (lo >> 24);
  lo &= 0xffffffu;
  // the 48-bit product is hi * 2^24 + lo; its top bit is bit 47 or 46
  bool top = hi >= 0x800000u;
  uint m = top ? hi : ((hi << 1) | (lo >> 23));
  uint guard = top ? (lo >> 23) : ((lo >> 22) & 1u);
  uint sticky = top ? ((lo & 0x7fffffu) != 0u ? 1u : 0u) : ((lo & 0x3fffffu) != 0u ? 1u : 0u);
  uint r = f_pack(s, e + (top ? 1 : 0), m, guard, sticky);
  bool za = ea == 0, zb = eb == 0, ia = aa == F_INF, ib = ab == F_INF;
  bool nan = aa > F_INF || ab > F_INF || (ia && zb) || (ib && za);
  r = (za || zb) ? (s << 31) : r;
  r = (ia || ib) ? ((s << 31) | F_INF) : r;
  return nan ? F_NAN : r;
}

uint f_div(uint a, uint b) {
  uint s = (a ^ b) >> 31;
  uint aa = a & 0x7fffffffu, ab = b & 0x7fffffffu;
  int ea = int(aa >> 23), eb = int(ab >> 23);
  int e = ea - eb + 127;
  uint ma = (aa & 0x7fffffu) | 0x800000u, mb = (ab & 0x7fffffu) | 0x800000u;
  // q = floor(r0 * 2^24 / mb) for r0 = ma, or twice ma when that is below mb (so r0 / mb is in [1, 2)): 25 bits, and the remainder says whether the quotient is exact.
  // The GPU's own division gives the quotient to within a few units; the exact remainder, taken modulo 2^32 (it is far smaller than that), then corrects it.
  bool less = ma < mb;
  uint r0 = less ? ma << 1 : ma;
  e -= less ? 1 : 0;
  int q = int(float(r0) / float(mb) * 16777216.0);
  int imb = int(mb);
  int rem = int((r0 << 24) - uint(q) * mb);
  int dq = int(floor(float(rem) / float(mb)));
  q += dq; rem -= dq * imb;
  for (int i = 0; i < 2; i++) {
    if (rem < 0) { q -= 1; rem += imb; }
    else if (rem >= imb) { q += 1; rem -= imb; }
  }
  uint qu = uint(q);
  uint res = f_pack(s, e, qu >> 1, qu & 1u, rem != 0 ? 1u : 0u);
  bool za = ea == 0, zb = eb == 0, ia = aa == F_INF, ib = ab == F_INF;
  bool nan = aa > F_INF || ab > F_INF || (ia && ib) || (za && zb);
  res = (ib || za) ? (s << 31) : res;
  res = (ia || zb) ? ((s << 31) | F_INF) : res;
  return nan ? F_NAN : res;
}

// float(n) for |n| < 2^24
uint f_from_int(int n) {
  uint s = n < 0 ? 1u : 0u;
  uint m = uint(n < 0 ? -n : n);
  int msb = f_msb(m | 1u);
  uint r = (s << 31) | (uint(msb + 127) << 23) | ((m << uint(23 - msb)) & 0x7fffffu);
  return n == 0 ? 0u : r;
}

// the spec's trunc: toward zero; -2^31 when |x| >= 2^31 or not a number
int f_trunc(uint a) {
  int e = int((a >> 23) & 255u) - 127;
  uint m = (a & 0x7fffffu) | 0x800000u;
  uint v = e >= 23 ? (m << uint(clamp(e - 23, 0, 31))) : (m >> uint(clamp(23 - e, 0, 31)));
  int r = (a >> 31) == 1u ? -int(v) : int(v);
  r = e < 0 ? 0 : r;
  return e >= 31 ? -2147483647 - 1 : r;
}

// ordered key: integers compare like the floats (zeros equal); not-a-numbers must be excluded by the caller
uint f_key(uint a) {
  uint k = (a >> 31) == 1u ? ~a : (a | 0x80000000u);
  return (a << 1) == 0u ? 0x80000000u : k;
}
bool f_isnan(uint a) { return (a & 0x7fffffffu) > F_INF; }
bool f_lt(uint a, uint b) { return (f_key(a) < f_key(b)) && !f_isnan(a) && !f_isnan(b); }
bool f_gt(uint a, uint b) { return f_lt(b, a); }
bool f_le(uint a, uint b) { return (f_key(a) <= f_key(b)) && !f_isnan(a) && !f_isnan(b); }
bool f_ge(uint a, uint b) { return f_le(b, a); }
bool f_eq(uint a, uint b) { return (f_key(a) == f_key(b)) && !f_isnan(a) && !f_isnan(b); }
bool f_iszero(uint a) { return (a << 1) == 0u; }
`,Et=`
uniform uint uFence;
uint f_ftz(uint a) { return (a & 0x7f800000u) == 0u ? (a & 0x80000000u) : a; }
uint f_fence(float x) { return floatBitsToUint(x) ^ uFence; }
uint f_add(uint a, uint b) {
  uint A = f_ftz(a), B = f_ftz(b);
  uint r = f_fence(uintBitsToFloat(A) + uintBitsToFloat(B));
  uint mA = A & 0x7fffffffu, mB = B & 0x7fffffffu;
  uint z = (mA | mB) == 0u ? (A & B & 0x80000000u) : (A == (B ^ 0x80000000u) ? 0u : ((mA >= mB ? A : B) & 0x80000000u));
  r = (r & 0x7f800000u) == 0u ? z : r;
  return (r & 0x7fffffffu) > F_INF ? F_NAN : r;
}
uint f_mul(uint a, uint b) {
  uint A = f_ftz(a), B = f_ftz(b);
  float fa = uintBitsToFloat(A), fb = uintBitsToFloat(B);
  uint r = f_fence(fa * fb);
  uint p2 = f_fence(uintBitsToFloat(f_fence(fa * 16777216.0)) * fb) & 0x7fffffffu;
  bool zero = (A & 0x7fffffffu) == 0u || (B & 0x7fffffffu) == 0u;
  uint tiny = ((A ^ B) & 0x80000000u) | (!zero && p2 >= 0x0c800000u ? 0x00800000u : 0u);
  r = (r & 0x7fffffffu) <= 0x00800000u ? tiny : r;
  return (r & 0x7fffffffu) > F_INF ? F_NAN : r;
}
`,Fi="uint f_add(uint a, uint b) {",Tt="uint f_sub(uint a, uint b)",Ei="uint f_mul(uint a, uint b) {",Ti="uint f_div(uint a, uint b)",St=(e,t,i)=>{const n=e.indexOf(t),s=e.indexOf(i);if(n<0||s<n)throw new Error("glsl-float: no "+t);return[e.slice(0,n)+e.slice(s),e.slice(n,s)]};function Nt(e){if(e==="soft")return Ft;const[t,i]=St(Ft,Fi,Tt),[n,s]=St(t,Ei,Ti),u=n.indexOf(Tt);if(e==="fast")return n.slice(0,u)+Et+n.slice(u);const r=(o,a)=>o.replace(/\bf_(add|mul)\(/g,a+"_$1("),f=r(i,"sf")+r(s,"sf")+r(Et,"ff")+`
bool gFloatMismatch = false;
uint f_add(uint a, uint b) { uint s = sf_add(a, b); gFloatMismatch = gFloatMismatch || s != ff_add(a, b); return s; }
uint f_mul(uint a, uint b) { uint s = sf_mul(a, b); gFloatMismatch = gFloatMismatch || s != ff_mul(a, b); return s; }
`;return n.slice(0,u)+f+n.slice(u)}const Y=5;class pt{words=new Uint32Array(4096*16);ints=new Int32Array(this.words.buffer);floats=new Float32Array(this.words.buffer);n=0;ensure(t){if(this.n+t<=this.words.length)return;let i=this.words.length;for(;i<this.n+t;)i*=2;const n=new Uint32Array(i);n.set(this.words.subarray(0,this.n)),this.words=n,this.ints=new Int32Array(n.buffer),this.floats=new Float32Array(n.buffer)}}function Rt(e,t,i){const n=e.ints,s=e.words;n[t]=F.minx,n[t+1]=F.maxx,n[t+2]=F.miny,n[t+3]=F.maxy,n[t+4]=F.DX12,n[t+5]=F.DY12,n[t+6]=F.DX23,n[t+7]=F.DY23,n[t+8]=F.DX31,n[t+9]=F.DY31,n[t+10]=F.C1,n[t+11]=F.C2,n[t+12]=F.C3,n[t+13]=F.xi,n[t+14]=F.yi,s[t+15]=At(F.ox),s[t+16]=At(F.oy),s[t+17]=i,s[t+18]=F.np,s[t+19]=0}class Ot{pool=new ci;trip=[];need=new Uint8Array(4096);lineIdx=[];run(t,i,n,s,u){const r=this.pool;r.clear();const f=li(t,i),o=this.trip;o.length=0;for(let l=0;l<n.length;l++)hi(n[l].type,n[l].count,n[l].base,o);this.need.length<i.n&&(this.need=new Uint8Array(Math.max(i.n,this.need.length*2)));const a=this.need;a.fill(0,0,i.n);for(let l=0;l<o.length;l++)a[o[l]]=1;const c=this.lineIdx;c.length=0;for(let l=0;l<n.length;l++){const b=n[l];if(b.type===5)for(let m=0;m+1<b.count;m+=2)c.push(b.base+m,b.base+m+1);else if(b.type===6)for(let m=0;m+1<b.count;m++)c.push(b.base+m,b.base+m+1)}for(let l=0;l<c.length;l++)a[c[l]]=1;const h=mi(f,t,i,r,a),p=pi(t.bp),x=E(t.bp[0],14,2),g=r.n,d=di(p),w=(Y+d)*4;for(let l=0;l+2<o.length;l+=3){if(r.n=g,!bi(r,x,h+o[l]*K,h+o[l+1]*K,h+o[l+2]*K))continue;const b=gi.n;for(let m=1;m+1<b;m++){u.ensure(w);const A=u.n;!_t(r,f,p,lt[0],lt[m],lt[m+1],u.floats,A+Y*4,null,4)||(Rt(u,A,s),u.n+=w)}}if(c.length){const l=E(t.bp[34],0,8);for(let b=0;b+1<c.length;b+=2){r.n=g;const m=xi(r,h+c[b]*K,h+c[b+1]*K);if(!m)continue;const A=[];wi(r,f,m[0],m[1],l,A);for(const _ of A){u.ensure(w);const k=u.n;!_t(r,f,p,_.v[0],_.v[1],_.v[2],u.floats,k+Y*4,_.scr,4)||(Rt(u,k,s),u.n+=w)}}}}}function Si(e,t){if(E(e[0],19,1)===1)return!0;for(const i of t)if(i.type===5||i.type===6)return!0;return!1}const D=4,yt=1<<D,Lt=640,Ri=528,Bi=Lt/yt;class jt{a=new Int32Array(1<<15);n=0;ensure(t){if(this.n+t<=this.a.length)return;let i=this.a.length;for(;i<this.n+t;)i*=2;const n=new Int32Array(i);n.set(this.a.subarray(0,this.n)),this.a=n}}function dt(e,t,i,n,s){let u=0;for(let r=0;r<i;r+=(Y+t[r+18])*4){u++;const f=e[r],o=e[r+1],a=e[r+2],c=e[r+3],h=Math.max(0,f)>>D,p=Math.max(0,a)>>D,x=Math.min(Lt-1,o)>>D,g=Math.min(Ri-1,c)>>D,d=e[r+4],w=e[r+5],l=e[r+6],b=e[r+7],m=e[r+8],A=e[r+9],_=e[r+10],k=e[r+11],S=e[r+12],v=Math.abs(d)<16384&&Math.abs(w)<16384&&Math.abs(l)<16384&&Math.abs(b)<16384&&Math.abs(m)<16384&&Math.abs(A)<16384&&Math.abs(_)<1073741824&&Math.abs(k)<1073741824&&Math.abs(S)<1073741824;if(x<h||g<p)continue;s.ensure(2*(x-h+1)*(g-p+1));const C=s.a;let L=s.n;const $=(r>>2)+n;for(let j=p;j<=g;j++)for(let Z=h;Z<=x;Z++){let ct=0;for(let Q=0;Q<2;Q++){const W=(j<<D)+(Q<<3),tt=W+7;if(!(tt<a||W>c))for(let it=0;it<2;it++){const V=(Z<<D)+(it<<3),et=V+7;et<f||V>o||v&&(_+16*(d>0?d*tt:d*W)-16*(w>0?w*V:w*et)<=0||k+16*(l>0?l*tt:l*W)-16*(b>0?b*V:b*et)<=0||S+16*(m>0?m*tt:m*W)-16*(A>0?A*V:A*et)<=0)||(ct|=1<<Q*2+it)}}ct!==0&&(C[L++]=j*Bi+Z,C[L++]=$|ct<<24)}s.n=L}return u}const Gt=25165824,Xt=26214400,qt=262144,Ui=Xt+qt,y=4096,z=288,Mi=`// 2^x for a binary32 x in [-8, 0], correctly rounded to binary32 (spec 11.3: the reference rounds a double-precision power once). The GPU's own exp2 is an
// approximation that is off in the last bit for about one argument in 300,000, which can move a fog level, so the power is computed in 64-bit fixed point
// from 16-bit limbs: |x| = n + t (t in [0, 1)); 2^-t is the product of 2^(-2^-k) for the first six bits of t, and of 2^-b for the rest b (< 2^-6), a Taylor
// series in z = b ln 2. The error is about 2^-60 of 1; the result's 24 bits and the 40 bits after them decide the rounding.
uniform uvec4 uExpC[8];     // 2^(-2^-k) for k = 1..6, ln 2, and in [7].x the number of Taylor terms (8); each a fraction of four 16-bit limbs, most significant first

// the high 64 bits of the product of two such fractions
uvec4 fxMul(uvec4 A, uvec4 B) {
  uint h0 = (A.x * B.x) >> 16;
  uint l0 = (A.x * B.x) & 0xffffu;
  uint p01 = A.x * B.y, p10 = A.y * B.x;
  uint h1 = (p01 >> 16) + (p10 >> 16), l1 = (p01 & 0xffffu) + (p10 & 0xffffu);
  uint p02 = A.x * B.z, p11 = A.y * B.y, p20 = A.z * B.x;
  uint h2 = (p02 >> 16) + (p11 >> 16) + (p20 >> 16), l2 = (p02 & 0xffffu) + (p11 & 0xffffu) + (p20 & 0xffffu);
  uint p03 = A.x * B.w, p12 = A.y * B.z, p21 = A.z * B.y, p30 = A.w * B.x;
  uint h3 = (p03 >> 16) + (p12 >> 16) + (p21 >> 16) + (p30 >> 16), l3 = (p03 & 0xffffu) + (p12 & 0xffffu) + (p21 & 0xffffu) + (p30 & 0xffffu);
  uint p13 = A.y * B.w, p22 = A.z * B.z, p31 = A.w * B.y;
  uint h4 = (p13 >> 16) + (p22 >> 16) + (p31 >> 16);
  uint a4 = l3 + h4;
  uint a3 = l2 + h3 + (a4 >> 16);
  uint a2 = l1 + h2 + (a3 >> 16);
  uint a1 = l0 + h1 + (a2 >> 16);
  uint a0 = h0 + (a1 >> 16);
  return uvec4(a0 & 0xffffu, a1 & 0xffffu, a2 & 0xffffu, a3 & 0xffffu);
}

uvec4 fxAdd(uvec4 A, uvec4 B) {
  uint s3 = A.w + B.w; uint s2 = A.z + B.z + (s3 >> 16); uint s1 = A.y + B.y + (s2 >> 16); uint s0 = A.x + B.x + (s1 >> 16);
  return uvec4(s0 & 0xffffu, s1 & 0xffffu, s2 & 0xffffu, s3 & 0xffffu);
}

// A - B for A >= B
uvec4 fxSub(uvec4 A, uvec4 B) {
  int d3 = int(A.w) - int(B.w); int c3 = d3 < 0 ? 1 : 0; d3 += c3 << 16;
  int d2 = int(A.z) - int(B.z) - c3; int c2 = d2 < 0 ? 1 : 0; d2 += c2 << 16;
  int d1 = int(A.y) - int(B.y) - c2; int c1 = d1 < 0 ? 1 : 0; d1 += c1 << 16;
  int d0 = int(A.x) - int(B.x) - c1;
  return uvec4(uint(d0) & 0xffffu, uint(d1), uint(d2), uint(d3));
}

uvec4 fxDiv(uvec4 A, uint k) {
  uint c = A.x; uint x = c / k; uint r = c - x * k;
  c = (r << 16) | A.y; uint y = c / k; r = c - y * k;
  c = (r << 16) | A.z; uint z = c / k; r = c - z * k;
  c = (r << 16) | A.w; uint w = c / k;
  return uvec4(x, y, z, w);
}

uint f_exp2(uint x) {
  uint ex = (x >> 23) & 255u;
  // |x| < 2^-25 (and zero): the power rounds to 1
  if (ex < 102u) return 0x3f800000u;
  uint m = (x & 0x7fffffu) | 0x800000u;
  int q = 150 - int(ex);                       // |x| = m * 2^-q
  uint n = ex >= 127u ? (m >> uint(q)) : 0u;   // the whole part
  uint mf = m & (q >= 32 ? 0xffffffffu : ((1u << uint(q)) - 1u));
  uint hi = q <= 32 ? (mf << uint(32 - q)) : (mf >> uint(q - 32));
  uint lo = q <= 32 ? 0u : (mf << uint(64 - q));
  if (hi == 0u && lo == 0u) return (127u - n) << 23;
  uint top6 = hi >> 26;
  uint hb = hi & 0x03ffffffu;
  uvec4 b = uvec4(hb >> 16, hb & 0xffffu, lo >> 16, lo & 0xffffu);
  uvec4 P = uvec4(0xffffu);
  for (int k = 0; k < 6; k++) if (((top6 >> uint(5 - k)) & 1u) == 1u) P = fxMul(P, uExpC[k]);
  uvec4 z = fxMul(b, uExpC[6]);
  uvec4 p = z, D = z;
  for (uint k = 2u; k <= uExpC[7].x; k++) {
    p = fxDiv(fxMul(p, z), k);
    D = (k & 1u) == 0u ? fxSub(D, p) : fxAdd(D, p);
  }
  uvec4 V = fxSub(P, fxMul(P, D));
  uint mant = (V.x << 8) | (V.y >> 8);                 // 24 bits, the top one set
  uint rest = ((V.y & 0xffu) << 24) | (V.z << 8) | (V.w >> 8);   // the next 32 bits (the last 8 of the 64 do not matter for the rounding)
  if (rest > 0x80000000u || (rest == 0x80000000u && (mant & 1u) == 1u)) mant += 1u;
  uint ee = 126u - n;
  if (mant == 0x1000000u) { mant = 0x800000u; ee += 1u; }
  return (ee << 23) | (mant & 0x7fffffu);
}
`;function G(e={}){const t=`#version 300 es
${e.indirect?"#define INDIRECT 1":""}
${e.fogExtra?"#define FOG_EXTRA 1":""}
${e.debugRegs?`#define DEBUG_REGS 1
#define DEBUG_A ${e.debugRegs[0]}
#define DEBUG_B ${e.debugRegs[1]}`:""}
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D uMem;
uniform usampler2D uData;   // RGBA32UI, 1024 texels wide
uniform usampler2D uTiles;
uniform usampler2D uList;
uniform usampler2D uStates;
uniform usampler2D uEfb;
uniform int uTilesX;
out uvec4 o;

#define TMEM_BASE ${Gt}u
#define HEADER_TEXELS ${Y}u
#define STATE_WORDS ${z}u
#define NO_PAL 0xffffffffu

${Nt(e.floatOps??"soft")}

uint wd(usampler2D t, uint i) { return texelFetch(t, ivec2(int(i & 4095u), int(i >> 12)), 0).r; }
// texel t of the triangle data (4 words)
uvec4 td(uint t) { return texelFetch(uData, ivec2(int(t & 1023u), int(t >> 10)), 0); }
uint mbyte(uint a) { return texelFetch(uMem, ivec2(int(a & 4095u), int(a >> 12)), 0).r; }
uint bf(uint v, int s, int n) { return (v >> uint(s)) & ((1u << uint(n)) - 1u); }
int bfi(uint v, int s, int n) { return int(bf(v, s, n)); }
int sx11(uint v) { return int(v << 21) >> 21; }
int idiv(int a, int b) { int q = abs(a) / b; return a < 0 ? -q : q; }

uint sbase;
uint S(uint k) { return wd(uStates, sbase + k); }

// ---------------------------------------------------------------------------------------------------------------------
// texture unit (spec 9)

struct MapI {
  uint fmt; int W; int H; int ws; int wt; uint pf; bool tmem; uint base; uint l1; uint l2; uint pal; uint palLen; uint mode0; uint mode1;
};

MapI loadMap(uint m) {
  bool hi = m >= 4u;
  uint k = m & 3u;
  uint img0 = S((hi ? 0xa8u : 0x88u) + k), img1 = S((hi ? 0xacu : 0x8cu) + k), img2 = S((hi ? 0xb0u : 0x90u) + k);
  uint img3 = S((hi ? 0xb4u : 0x94u) + k), tl = S((hi ? 0xb8u : 0x98u) + k);
  MapI mp;
  mp.fmt = bf(img0, 20, 4);
  mp.W = bfi(img0, 0, 10);
  mp.H = bfi(img0, 10, 10);
  uint m0 = S((hi ? 0xa0u : 0x80u) + k);
  mp.mode0 = m0;
  mp.mode1 = S((hi ? 0xa4u : 0x84u) + k);
  mp.ws = bfi(m0, 0, 2);
  mp.wt = bfi(m0, 2, 2);
  mp.pf = bf(tl, 10, 2);
  mp.tmem = bf(img1, 21, 1) == 1u;
  mp.base = (img3 & 0xffffffu) * 32u;
  mp.l1 = bf(img1, 0, 15) * 32u;
  mp.l2 = bf(img2, 0, 15) * 32u;
  mp.pal = S(264u + m);
  mp.palLen = S(274u + m);
  return mp;
}

// address in uMem of byte i of the image (TMEM images live in the TMEM area; an RGBA8 image keeps its green/blue halves in the other bank)
uint tAddr(MapI mp, uint i) {
  if (mp.tmem) {
    if (mp.fmt == 6u) {
      uint blk = i >> 6, r = i & 63u;
      return TMEM_BASE + ((r < 32u ? mp.l1 + blk * 32u + r : mp.l2 + blk * 32u + r - 32u) & 0xfffffu);
    }
    return TMEM_BASE + ((mp.l1 + i) & 0xfffffu);
  }
  return (mp.base + i) & 0x1ffffffu;
}
uint tb(MapI mp, uint i) { return mbyte(tAddr(mp, i)); }

int w3(int v) { return (v << 5) | (v << 2) | (v >> 1); }
int w4(int v) { return v * 17; }
int w5(int v) { return (v << 3) | (v >> 2); }
int w6(int v) { return (v << 2) | (v >> 4); }

ivec4 rgb5a3(uint v) {
  if ((v & 0x8000u) != 0u) return ivec4(w5(bfi(v, 10, 5)), w5(bfi(v, 5, 5)), w5(bfi(v, 0, 5)), 255);
  return ivec4(w4(bfi(v, 8, 4)), w4(bfi(v, 4, 4)), w4(bfi(v, 0, 4)), w3(bfi(v, 12, 3)));
}

ivec4 palColor(uint v, uint pf) {
  if (pf == 0u) { int i = int(v & 255u); return ivec4(i, i, i, int(v >> 8)); }
  if (pf == 1u) return ivec4(w5(bfi(v, 11, 5)), w6(bfi(v, 5, 6)), w5(bfi(v, 0, 5)), 255);
  return rgb5a3(v);
}

void finfo(uint fmt, out int kw, out int kh, out uint bytes) {
  bytes = 32u;
  if (fmt == 0u || fmt == 8u || fmt == 14u) { kw = 3; kh = 3; }
  else if (fmt == 1u || fmt == 2u || fmt == 9u) { kw = 3; kh = 2; }
  else if (fmt == 6u) { kw = 2; kh = 2; bytes = 64u; }
  else { kw = 2; kh = 2; }
}

// half bytes per texel, for the mip offsets (spec 9.3)
int nibblesOf(uint fmt) {
  if (fmt == 0u || fmt == 8u || fmt == 14u) return 1;
  if (fmt == 1u || fmt == 2u || fmt == 9u) return 2;
  if (fmt == 6u) return 8;
  return 4;
}

// one texel of a level (wrapped coordinates s, t), spec 9.2. Every format reads at most five bytes through the same five reads.
ivec4 fetchTexel(MapI mp, uint off, int W, int s, int t) {
  int kw, kh; uint bytes;
  finfo(mp.fmt, kw, kh, bytes);
  int bpr = (W >> kw) + 1;
  int blk = (t >> kh) * bpr + (s >> kw);
  uint base = off + uint(blk) * bytes;
  int tx = s & ((1 << kw) - 1), ty = t & ((1 << kh) - 1);
  uint fmt = mp.fmt;
  uint a0 = base, d2 = 2u, a4 = 0u;
  if (fmt == 0u || fmt == 8u) a0 = base + uint((ty * 8 + tx) >> 1);
  else if (fmt == 1u || fmt == 9u || fmt == 2u) a0 = base + uint(ty * 8 + tx);
  else if (fmt == 6u) { a0 = base + uint((ty * 4 + tx) * 2); d2 = 32u; }
  else if (fmt == 14u) { a0 = base + uint(((ty >> 2) << 1) + (tx >> 2)) * 8u; a4 = a0 + 4u + uint(ty & 3); }
  else a0 = base + uint((ty * 4 + tx) * 2);
  uint b0 = tb(mp, a0), b1 = tb(mp, a0 + 1u), b2 = 0u, b3 = 0u, b4 = 0u;
  if (fmt == 6u || fmt == 14u) { b2 = tb(mp, a0 + d2); b3 = tb(mp, a0 + d2 + 1u); }
  if (fmt == 14u) b4 = tb(mp, a4);
  uint v16 = (b0 << 8) | b1;
  uint pidx = 0u;
  bool indexed = false;
  ivec4 r = ivec4(0);
  if (fmt == 0u) { int v = w4((tx & 1) == 1 ? int(b0 & 15u) : int(b0 >> 4)); r = ivec4(v); }
  else if (fmt == 8u) { pidx = (tx & 1) == 1 ? (b0 & 15u) : (b0 >> 4); indexed = true; }
  else if (fmt == 1u) r = ivec4(int(b0));
  else if (fmt == 9u) { pidx = b0; indexed = true; }
  else if (fmt == 2u) { int i = w4(int(b0 & 15u)); r = ivec4(i, i, i, w4(int(b0 >> 4))); }
  else if (fmt == 3u) { int i = int(b1); r = ivec4(i, i, i, int(b0)); }
  else if (fmt == 4u) r = ivec4(w5(bfi(v16, 11, 5)), w6(bfi(v16, 5, 6)), w5(bfi(v16, 0, 5)), 255);
  else if (fmt == 5u) r = rgb5a3(v16);
  else if (fmt == 10u) { pidx = v16 & 0x3fffu; indexed = true; }
  else if (fmt == 6u) r = ivec4(int(b1), int(b2), int(b3), int(b0));
  else if (fmt == 14u) {
    uint c1 = v16, c2 = (b2 << 8) | b3;
    int sel = int((b4 >> uint(6 - 2 * (tx & 3))) & 3u);
    ivec3 a = ivec3(w5(bfi(c1, 11, 5)), w6(bfi(c1, 5, 6)), w5(bfi(c1, 0, 5)));
    ivec3 b = ivec3(w5(bfi(c2, 11, 5)), w6(bfi(c2, 5, 6)), w5(bfi(c2, 0, 5)));
    if (sel == 0) r = ivec4(a, 255);
    else if (sel == 1) r = ivec4(b, 255);
    else if (c1 > c2) {
      ivec3 d = b - a;
      ivec3 th = ivec3(0);
      for (int c = 0; c < 3; c++) { uint du = uint(d[c]); th[c] = int((du >> 1) - (du >> 3)); }
      r = sel == 2 ? ivec4((a + th) & 255, 255) : ivec4((b - th) & 255, 255);
    } else r = ivec4((a + b + 1) >> 1, sel == 2 ? 255 : 0);
  }
  if (indexed) {
    uint pe = 0u;
    if (mp.pal != NO_PAL && pidx * 2u + 1u < mp.palLen) pe = (mbyte(mp.pal + pidx * 2u) << 8) | mbyte(mp.pal + pidx * 2u + 1u);
    r = palColor(pe, mp.pf);
  }
  return r;
}

int wrapC(int c, int Sz, int mode) {
  if (mode == 0) return min(max(c, 0), Sz);
  int n = Sz + 1;
  if (mode == 1) { int q = idiv(c, n); int r = c - q * n; return r < 0 ? Sz + r : r; }
  int q = idiv(c, n);
  int r = abs(c - q * n);
  return (q & 1) != 0 ? Sz - r : r;
}

uint levelOffset(uint fmt, int W, int H, int L) {
  int kw, kh; uint bytes;
  finfo(fmt, kw, kh, bytes);
  int bw = 1 << kw, bh = 1 << kh;
  int nib = nibblesOf(fmt);
  uint off = 0u;
  int w = W + 1, h = H + 1;
  for (int l = 1; l <= L; l++) {
    off += uint((max(w, bw) * max(h, bh) * nib) >> 1);
    w >>= 1; h >>= 1;
  }
  return off;
}

// the unit's sample (spec 9.3). Loops over the mip levels and the filter taps have run-time bounds so that the compiler keeps one
// copy of the texel fetch.
ivec4 sampleMap(MapI mp, int s, int t, int lod, bool linear) {
  uint mf = bf(mp.mode0, 5, 3);
  int fract = lod & 15;
  int mip = 0;
  bool mixl = false;
  if (lod > 0 && (mf & 3u) != 0u) {
    mip = lod >> 4;
    mixl = fract != 0 && (mf & 2u) != 0u;
    mip += (fract >> 3) & int(mf & 1u);
  }
  int nlev = mixl ? 2 : 1;
  int ntaps = linear ? 4 : 1;
  ivec4 acc0 = ivec4(0), acc1 = ivec4(0);
  for (int li = 0; li < uK.w; li++) {
    if (li >= nlev) break;
    int L = mip + li;
    uint off = L > 0 ? levelOffset(mp.fmt, mp.W, mp.H, L) : 0u;
    int W = mp.W >> L, H = mp.H >> L;
    int sl = s >> L, tl = t >> L;
    int s0 = linear ? (sl - 64) >> 7 : sl >> 7, t0 = linear ? (tl - 64) >> 7 : tl >> 7;
    int fs = (sl - 64) & 127, ft = (tl - 64) & 127;
    ivec4 sum = ivec4(0);
    for (int k = 0; k < uK.x; k++) {
      if (k >= ntaps) break;
      int dx = k & 1, dy = k >> 1;
      int wgt = linear ? ((dx == 1 ? fs : 128 - fs) * (dy == 1 ? ft : 128 - ft)) : 1;
      sum += fetchTexel(mp, off, W, wrapC(s0 + dx, W, mp.ws), wrapC(t0 + dy, H, mp.wt)) * wgt;
    }
    if (linear) sum = sum >> 14;
    if (li == 0) acc0 = sum; else acc1 = sum;
  }
  return mixl ? (acc0 * (16 - fract) + acc1 * fract) >> 4 : acc0;
}

// ---------------------------------------------------------------------------------------------------------------------
// planes and texture coordinates (spec 7.4, 8.3, 8.4)

uint gTri;      // word offset of the current triangle in uData
uint gNp;       // number of planes of the current triangle
uint gState;    // its draw state
uint gOx, gOy;  // float bits
int gXi, gYi;

uint planeAt(uint k, uint dx, uint dy) {
  // a plane the triangle does not carry (a generator past BP's count) reads zero
  if (k >= gNp) return 0u;
  uvec4 p = td(gTri + HEADER_TEXELS + k);
  // a plane with no slope (both differences zero) is its own constant: adding the zero products changes nothing in a finite non-zero value
  uint ex = (p.x >> 23) & 255u;
  if ((p.y & 0x7f800000u) == 0u && (p.z & 0x7f800000u) == 0u && ex != 0u && ex != 255u) return p.x;
  return f_add(f_add(p.x, f_mul(p.y, dx)), f_mul(p.z, dy));
}

// s and t of generator n at pixel (px, py)
void genST(uint n, int px, int py, out uint s, out uint t) {
  uint dx = f_add(gOx, f_from_int(px - gXi)), dy = f_add(gOy, f_from_int(py - gYi));
  // the four planes (1/w, u, v, w) through one copy of the evaluation
  uint pv[4];
  bool projective = ((S(272u) >> n) & 1u) == 1u;
  int nPlanes = projective ? uK.x : uK.x - 1;
  for (int k = 0; k < nPlanes; k++) pv[k] = planeAt(k == 0 ? 0u : 9u + n * 3u + uint(k), dx, dy);
  uint iw = f_div(F_ONE, pv[0]);
  uint proj = iw;
  if (projective) {
    uint q = f_mul(pv[3], iw);
    proj = f_iszero(q) ? iw : f_div(iw, q);
  }
  s = f_mul(pv[1], proj);
  t = f_mul(pv[2], proj);
}

uint f_max(uint a, uint b) { return f_lt(a, b) ? b : a; }

int log2fix(uint d) { return int(((d & 0x7f800000u) >> 19) + ((d & 0x007fffffu) >> 19)) - 2032; }

// the texture stage's coordinates: the pixel's s and t of generator n (7 fractional bits) and the level of detail map mp gives (8.3, 8.4)
void stageCoords(MapI mp, uint n, int x, int y, out int S24, out int T24, out int lod, out bool linear) {
  int bx = x & ~1, by = y & ~1;
  int me = (x & 1) + ((y & 1) << 1);
  int maxl = bfi(mp.mode1, 8, 8), minl = bfi(mp.mode1, 0, 8);
  bool magLin = bf(mp.mode0, 4, 1) == 1u;
  uint mf = bf(mp.mode0, 5, 3);
  // When the clamp range of the level of detail is one value and the magnification and minification filters agree on linear or not, neither the level of
  // detail nor the filter depends on the differences across the block (the clamp gives that one value for every lod, and 'linear' is the same on both
  // sides of lod 0): the pixel's own coordinates are all there is to compute.
  bool constLod = maxl == minl && magLin == ((mf & 4u) != 0u);
  uint sv[4], tv[4];
  for (int j = 0; j < uK.x; j++) {
    sv[j] = 0u; tv[j] = 0u;
    if (!constLod || j == me) genST(n, bx + (j & 1), by + (j >> 1), sv[j], tv[j]);
  }
  if (constLod) {
    lod = minl; linear = magLin;
  } else {
    // the largest change of s and of t across the block: one copy of the differences for both and for the three neighbors
    uint dm[2];
    bool diag = bf(mp.mode0, 8, 1) == 1u;
    for (int m = 0; m < uK.w; m++) {
      uint c0 = m == 0 ? sv[0] : tv[0];
      uint dd[3];
      for (int k = 0; k < uK.x - 1; k++) {
        uint ck = m == 0 ? (k == 0 ? sv[1] : (k == 1 ? sv[2] : sv[3])) : (k == 0 ? tv[1] : (k == 1 ? tv[2] : tv[3]));
        dd[k] = f_abs(f_sub(c0, ck));
      }
      dm[m] = diag ? dd[2] : f_max(dd[0], dd[1]);
    }
    uint d = f_max(dm[0], dm[1]);
    int bias = int(bf(mp.mode0, 9, 8) << 24) >> 24;
    lod = log2fix(d) + (bias >> 1);
    linear = (lod > 0 && (mf & 4u) != 0u) || (lod <= 0 && magLin);
    if (lod > maxl) lod = maxl; else if (lod < minl) lod = minl;
  }
  int st[2];
  for (int c = 0; c < uK.w; c++) st[c] = (f_trunc(f_mul(c == 0 ? sv[me] : tv[me], 0x43000000u)) << 8) >> 8;
  S24 = st[0];
  T24 = st[1];
}

#ifdef INDIRECT
// the offsets and the alpha bump an indirect sample gives a TEV stage (spec 10.5): (cs, ct) are the stage's own coordinates, tx the
// indirect texture's sample; the stage's coordinates become (oS, oT)
void indirectOffsets(uint ind, ivec4 tx, int cs, int ct, int prevS, int prevT, out int oS, out int oT, out int bump) {
  int fmt = bfi(ind, 2, 2), biasBits = bfi(ind, 4, 3), bs = bfi(ind, 7, 2), mid = bfi(ind, 9, 4);
  int sw = bfi(ind, 13, 3), tw = bfi(ind, 16, 3);
  ivec3 i = ivec3(tx.a, tx.b, tx.g);
  int mask = fmt == 1 ? 0xe0 : (fmt == 2 ? 0xf0 : 0xf8);
  bump = bs == 0 ? 0 : (i[bs - 1] & mask);
  int vmask = fmt == 0 ? 0xff : (fmt == 1 ? 0x1f : (fmt == 2 ? 0x0f : 0x07));
  ivec3 v = i & vmask;
  for (int k = 0; k < 3; k++) if (((biasBits >> k) & 1) == 1) v[k] += fmt == 0 ? -128 : 1;
  int o0 = 0, o1 = 0;
  bool keep = false;
  int m = mid & 3;
  if (m != 0) {
    uint A = S(0x06u + 3u * uint(m - 1)), B = S(0x07u + 3u * uint(m - 1)), C = S(0x08u + 3u * uint(m - 1));
    int ma = sx11(A & 0x7ffu), mb = sx11((A >> 11) & 0x7ffu), mc = sx11(B & 0x7ffu), md = sx11((B >> 11) & 0x7ffu), me = sx11(C & 0x7ffu), mf = sx11((C >> 11) & 0x7ffu);
    int ex = bfi(A, 22, 2) | (bfi(B, 22, 2) << 2) | (bfi(C, 22, 2) << 4);
    int shift = 17 - ex;
    int type = mid & 12;
    // matrix type 12 is not defined: the stage's coordinate stays what the previous stage left (spec 10.5)
    if (type == 12) keep = true;
    else if (type == 0) { o0 = (ma * v.x + mc * v.y + me * v.z) >> 3; o1 = (mb * v.x + md * v.y + mf * v.z) >> 3; }
    else if (type == 4) { o0 = idiv(cs * v.x, 256); o1 = idiv(ct * v.x, 256); }
    else if (type == 8) { o0 = idiv(cs * v.y, 256); o1 = idiv(ct * v.y, 256); }
    // shift counts are taken modulo 32 (as the CPU reference's shifts do)
    if (shift >= 0) { o0 >>= (shift & 31); o1 >>= (shift & 31); } else { o0 <<= ((-shift) & 31); o1 <<= ((-shift) & 31); }
  }
  int ws = sw == 0 ? cs : (sw >= 6 ? 0 : (cs & (((256 >> (sw - 1)) << 7) - 1)));
  int wt = tw == 0 ? ct : (tw >= 6 ? 0 : (ct & (((256 >> (tw - 1)) << 7) - 1)));
  int rs = ws + o0, rt = wt + o1;
  if (bf(ind, 20, 1) == 1u) { rs += prevS; rt += prevT; }
  oS = keep ? prevS : (rs << 8) >> 8;
  oT = keep ? prevT : (rt << 8) >> 8;
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// TEV (spec 10)

int gRegs[16];
int gKk[16];
int gTex[4];
int gRas[4];
int gRasCh[8];
int gLastTex[4];

int clampTev(int r, int clampBit) {
  r = (r << 16) >> 16;
  if (clampBit == 1) return r < 0 ? 0 : (r > 255 ? 255 : r);
  return r < -1024 ? -1024 : (r > 1023 ? 1023 : r);
}

int tevRegular(bool isAlpha, int a, int b, int c, int d, int bias, int op, int scale, int clampBit) {
  int ls = scale == 1 ? 1 : (scale == 2 ? 2 : 0), rs = scale == 3 ? 1 : 0;
  int B = bias == 1 ? 128 : (bias == 2 ? -128 : 0);
  int cw = c + (c >> 7);
  int t = a * (256 - cw) + b * cw;
  t = t << ls;
  if (!isAlpha) {
    t = t + (scale == 3 ? 0 : (op == 1 ? 127 : 128));
    t = t >> 8;
    t = op == 1 ? -t : t;
  } else {
    t = t + (scale != 3 ? 0 : (op == 1 ? 127 : 128));
    t = op == 1 ? ((-t) >> 8) : (t >> 8);
  }
  int r = ((d + B) << ls) + t;
  r = r >> rs;
  return clampTev(r, clampBit);
}

ivec3 clampTev3(ivec3 r, int clampBit) {
  r = (r << 16) >> 16;
  return clampBit == 1 ? clamp(r, 0, 255) : clamp(r, -1024, 1023);
}

// the color combiner's regular arithmetic (10.4) on all three components at once
ivec3 tevColor3(ivec3 a, ivec3 b, ivec3 c, ivec3 d, int bias, int op, int scale, int clampBit) {
  int ls = scale == 1 ? 1 : (scale == 2 ? 2 : 0), rs = scale == 3 ? 1 : 0;
  int B = bias == 1 ? 128 : (bias == 2 ? -128 : 0);
  ivec3 cw = c + (c >> 7);
  ivec3 t = (a * (256 - cw) + b * cw) << ls;
  t = (t + (scale == 3 ? 0 : (op == 1 ? 127 : 128))) >> 8;
  t = op == 1 ? -t : t;
  return clampTev3((((d + B) << ls) + t) >> rs, clampBit);
}

// the compare modes (10.4) of the color combiner: a result per component
bvec3 compare3(int mode, ivec3 A, ivec3 B) {
  if (mode == 6) return greaterThan(A, B);
  if (mode == 7) return equal(A, B);
  bool r;
  if (mode == 0) r = A.x > B.x;
  else if (mode == 1) r = A.x == B.x;
  else if (mode == 2) r = ((A.y << 8) | A.x) > ((B.y << 8) | B.x);
  else if (mode == 3) r = ((A.y << 8) | A.x) == ((B.y << 8) | B.x);
  else if (mode == 4) r = ((A.z << 16) | (A.y << 8) | A.x) > ((B.z << 16) | (B.y << 8) | B.x);
  else r = ((A.z << 16) | (A.y << 8) | A.x) == ((B.z << 16) | (B.y << 8) | B.x);
  return bvec3(r);
}

bool compareCond(int mode, ivec3 A, ivec3 B, int k) {
  if (mode == 0) return A.x > B.x;
  if (mode == 1) return A.x == B.x;
  if (mode == 2) return ((A.y << 8) | A.x) > ((B.y << 8) | B.x);
  if (mode == 3) return ((A.y << 8) | A.x) == ((B.y << 8) | B.x);
  if (mode == 4) return ((A.z << 16) | (A.y << 8) | A.x) > ((B.z << 16) | (B.y << 8) | B.x);
  if (mode == 5) return ((A.z << 16) | (A.y << 8) | A.x) == ((B.z << 16) | (B.y << 8) | B.x);
  if (mode == 6) return A[k] > B[k];
  return A[k] == B[k];
}

int ksel8(int i) { return i == 0 ? 255 : (i == 1 ? 223 : (i == 2 ? 191 : (i == 3 ? 159 : (i == 4 ? 128 : (i == 5 ? 96 : (i == 6 ? 64 : 32)))))); }

ivec3 konstColor(int ks) {
  if (ks < 8) return ivec3(ksel8(ks));
  if (ks < 12) return ivec3(0);
  if (ks < 16) { int k = (ks - 12) * 4; return ivec3(gKk[k], gKk[k + 1], gKk[k + 2]); }
  int comp = (ks - 16) >> 2, k = (ks & 3) * 4;
  return ivec3(gKk[k + comp]);
}

int konstAlpha(int ks) {
  if (ks < 8) return ksel8(ks);
  if (ks < 16) return 0;
  int comp = (ks - 16) >> 2, k = (ks & 3) * 4;
  return gKk[k + comp];
}

ivec3 csel(int k) {
  switch (k) {
    case 0: return ivec3(gRegs[0], gRegs[1], gRegs[2]);
    case 1: return ivec3(gRegs[3]);
    case 2: return ivec3(gRegs[4], gRegs[5], gRegs[6]);
    case 3: return ivec3(gRegs[7]);
    case 4: return ivec3(gRegs[8], gRegs[9], gRegs[10]);
    case 5: return ivec3(gRegs[11]);
    case 6: return ivec3(gRegs[12], gRegs[13], gRegs[14]);
    case 7: return ivec3(gRegs[15]);
    case 8: return ivec3(gTex[0], gTex[1], gTex[2]);
    case 9: return ivec3(gTex[3]);
    case 10: return ivec3(gRas[0], gRas[1], gRas[2]);
    case 11: return ivec3(gRas[3]);
    case 12: return ivec3(255);
    case 13: return ivec3(128);
  }
  return ivec3(0);
}

int asel(int k, int kas) {
  switch (k) {
    case 0: return gRegs[3];
    case 1: return gRegs[7];
    case 2: return gRegs[11];
    case 3: return gRegs[15];
    case 4: return gTex[3];
    case 5: return gRas[3];
    case 6: return konstAlpha(kas);
  }
  return 0;
}

void swz(uint t, int v[4], out int o4[4]) {
  uint w0 = S(0xf6u + 2u * t), w1 = S(0xf7u + 2u * t);
  o4[0] = v[bfi(w0, 0, 2)]; o4[1] = v[bfi(w0, 2, 2)]; o4[2] = v[bfi(w1, 0, 2)]; o4[3] = v[bfi(w1, 2, 2)];
}

void loadColorRegs() {
  for (uint r = 0u; r < 4u; r++) {
    uint ra = S(0xe0u + 2u * r), bg = S(0xe1u + 2u * r);
    gRegs[int(r) * 4] = sx11(ra & 0x7ffu); gRegs[int(r) * 4 + 3] = sx11((ra >> 12) & 0x7ffu);
    gRegs[int(r) * 4 + 2] = sx11(bg & 0x7ffu); gRegs[int(r) * 4 + 1] = sx11((bg >> 12) & 0x7ffu);
    uint ka = S(256u + 2u * r), kb = S(257u + 2u * r);
    gKk[int(r) * 4] = sx11(ka & 0x7ffu); gKk[int(r) * 4 + 3] = sx11((ka >> 12) & 0x7ffu);
    gKk[int(r) * 4 + 2] = sx11(kb & 0x7ffu); gKk[int(r) * 4 + 1] = sx11((kb >> 12) & 0x7ffu);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// fog, blend

uint fogFloat(uint w) {
  return (((w >> 19) & 1u) << 31) | (((w >> 11) & 255u) << 23) | ((w & 0x7ffu) << 12);
}

#ifdef FOG_EXTRA
// correctly rounded square root of a non-negative float (the fog range adjustment): the integer square root of the mantissa scaled to
// 48 bits, found from the GPU's own estimate and corrected against exact products
uint f_sqrt(uint a) {
  int e = int(a >> 23) - 127;
  uint m = (a & 0x7fffffu) | 0x800000u;
  bool odd = (e & 1) != 0;
  uint M = odd ? m << 1 : m;
  e = odd ? e - 1 : e;
  // N = M * 2^23 as (hi, lo) of 24 bits each
  uint nh = M >> 1, nl = (M & 1u) << 23;
  uint s = uint(sqrt(float(M) * 8388608.0));
  s = clamp(s, 0x800000u, 0xffffffu);
  for (int i = 0; i < uK.z; i++) {
    // s * s as (hi, lo)
    uint sh = s >> 12, sl = s & 0xfffu;
    uint mid = 2u * sh * sl;
    uint lo = sl * sl + ((mid & 0xfffu) << 12);
    uint hi = sh * sh + (mid >> 12) + (lo >> 24);
    lo &= 0xffffffu;
    bool tooBig = hi > nh || (hi == nh && lo > nl);
    s = tooBig ? s - 1u : s;
    // (s + 1)^2
    uint t = s + 1u;
    uint th = t >> 12, tl = t & 0xfffu;
    uint tmid = 2u * th * tl;
    uint tlo = tl * tl + ((tmid & 0xfffu) << 12);
    uint thi = th * th + (tmid >> 12) + (tlo >> 24);
    tlo &= 0xffffffu;
    bool fits = thi < nh || (thi == nh && tlo <= nl);
    s = fits ? s + 1u : s;
  }
  // remainder N - s^2, to round to nearest: up when it exceeds s
  uint sh2 = s >> 12, sl2 = s & 0xfffu;
  uint mid2 = 2u * sh2 * sl2;
  uint lo2 = sl2 * sl2 + ((mid2 & 0xfffu) << 12);
  uint hi2 = sh2 * sh2 + (mid2 >> 12) + (lo2 >> 24);
  lo2 &= 0xffffffu;
  uint rem = ((nh - hi2) << 24) + nl - lo2;
  uint r = s + (rem > s ? 1u : 0u);
  bool carry = r == 0x1000000u;
  r = carry ? 0x800000u : r;
  int er = (e >> 1) + (carry ? 1 : 0) + 127;
  uint res = (uint(er) << 23) | (r & 0x7fffffu);
  return (a << 1) == 0u ? 0u : res;
}

${Mi}
#endif

void fogPixel(uint fsel, int z, int x, inout ivec4 c) {
  uint A = fogFloat(S(0xeeu)), C = fogFloat(S(0xf1u) & 0xfffffu);
  uint Bm = S(0xefu);
  int Bs = int(S(0xf0u) & 31u);
  uint ze;
  if (bf(S(0xf1u), 20, 1) == 0u) ze = f_div(f_mul(A, 0x4b7fffffu), f_from_int(int(Bm) - (z >> Bs)));
  else ze = f_mul(A, f_div(f_from_int(z), 0x4b7fffffu));
#ifdef FOG_EXTRA
  if (bf(S(0xe8u), 10, 1) == 1u) {
    uint o = f_div(f_from_int(x - (bfi(S(0xe8u), 0, 10) - 342)), S(273u));
    uint fidx = f_sub(0x41100000u, f_mul(f_abs(o), 0x41100000u));
    fidx = f_lt(fidx, 0u) ? 0u : (f_gt(fidx, 0x41100000u) ? 0x41100000u : fidx);
    int lo = f_trunc(fidx), up = lo + 1;
    uint wlo = S(0xe9u + uint(lo >> 1)), wup = S(0xe9u + uint(up >> 1));
    uint Klo = f_mul(f_div(f_from_int(int((lo & 1) == 1 ? (wlo & 0xfffu) : ((wlo >> 12) & 0xfffu))), 0x43800000u), 0x40800000u);
    uint Kup = f_mul(f_div(f_from_int(int((up & 1) == 1 ? (wup & 0xfffu) : ((wup >> 12) & 0xfffu))), 0x43800000u), 0x40800000u);
    uint factor = f_sub(f_from_int(up), fidx);
    uint k = f_add(f_mul(Klo, factor), f_mul(Kup, f_sub(F_ONE, factor)));
    ze = f_mul(ze, f_div(f_sqrt(f_add(f_mul(o, o), f_mul(k, k))), k));
  }
#endif
  ze = f_sub(ze, C);
  uint fog = f_lt(ze, 0u) ? 0u : (f_gt(ze, F_ONE) ? F_ONE : ze);
#ifdef FOG_EXTRA
  if (fsel == 4u) fog = f_sub(F_ONE, f_exp2(f_mul(0xc1000000u, fog)));
  else if (fsel == 5u) fog = f_sub(F_ONE, f_exp2(f_mul(f_mul(0xc1000000u, fog), fog)));
  else if (fsel == 6u) fog = f_exp2(f_mul(0xc1000000u, f_sub(F_ONE, fog)));
  else if (fsel == 7u) { uint g = f_sub(F_ONE, fog); fog = f_exp2(f_mul(f_mul(0xc1000000u, g), g)); }
#endif
  int fi = f_trunc(f_mul(fog, 0x43800000u));
  uint fc = S(0xf2u);
  ivec3 fr = ivec3(int((fc >> 16) & 255u), int((fc >> 8) & 255u), int(fc & 255u));
  c.rgb = (c.rgb * (256 - fi) + fr * fi) >> 8;
}

bool cmpOp(uint code, int a, int b) {
  switch (code) {
    case 0u: return false;
    case 1u: return a < b;
    case 2u: return a == b;
    case 3u: return a <= b;
    case 4u: return a > b;
    case 5u: return a != b;
    case 6u: return a >= b;
  }
  return true;
}

int blendFactor(uint code, int self, ivec4 s, ivec4 d, bool isDst) {
  switch (code) {
    case 0u: return 0;
    case 1u: return 255;
    case 2u: return isDst ? self : self;
    case 3u: return 255 - self;
    case 4u: return s.a;
    case 5u: return 255 - s.a;
    case 6u: return d.a;
  }
  return 255 - d.a;
}

ivec4 blendPixel(uint mode, uint dstAlpha, ivec4 s, ivec4 d) {
  ivec4 r;
  bool enable = (mode & 1u) == 1u, logic = ((mode >> 1) & 1u) == 1u, sub = ((mode >> 11) & 1u) == 1u;
  uint dfc = (mode >> 5) & 7u, sfc = (mode >> 8) & 7u, lop = (mode >> 12) & 15u;
  if (enable) {
    if (sub) r = max(ivec4(0), d - s);
    else {
      for (int k = 0; k < 4; k++) {
        int fs = blendFactor(sfc, d[k], s, d, false);
        int fd = blendFactor(dfc, s[k], s, d, true);
        fs += fs >> 7; fd += fd >> 7;
        r[k] = min(255, (s[k] * fs + d[k] * fd) >> 8);
      }
    }
  } else if (logic) {
    uint Sw = uint(s.a) | (uint(s.b) << 8) | (uint(s.g) << 16) | (uint(s.r) << 24);
    uint Dw = uint(d.a) | (uint(d.b) << 8) | (uint(d.g) << 16) | (uint(d.r) << 24);
    uint w;
    switch (lop) {
      case 0u: w = 0u; break;
      case 1u: w = Sw & Dw; break;
      case 2u: w = Sw & ~Dw; break;
      case 3u: w = Sw; break;
      case 4u: w = ~Sw & Dw; break;
      case 5u: w = Dw; break;
      case 6u: w = Sw ^ Dw; break;
      case 7u: w = Sw | Dw; break;
      case 8u: w = ~(Sw | Dw); break;
      case 9u: w = ~(Sw ^ Dw); break;
      case 10u: w = ~Dw; break;
      case 11u: w = Sw | ~Dw; break;
      case 12u: w = ~Sw; break;
      case 13u: w = ~Sw | Dw; break;
      case 14u: w = ~(Sw & Dw); break;
      default: w = 0xffffffffu;
    }
    r = ivec4(int(w >> 24), int((w >> 16) & 255u), int((w >> 8) & 255u), int(w & 255u));
  } else r = s;
  if (((dstAlpha >> 8) & 1u) == 1u) r.a = int(dstAlpha & 255u);
  return r;
}

ivec4 readColor(uint w, uint fmt) {
  if (fmt == 1u) {
    return ivec4(w6(bfi(w, 18, 6)), w6(bfi(w, 12, 6)), w6(bfi(w, 6, 6)), w6(bfi(w, 0, 6)));
  }
  return ivec4(int((w >> 16) & 255u), int((w >> 8) & 255u), int(w & 255u), 255);
}

uint writeColor(uint old, uint fmt, uint mode, ivec4 r) {
  bool colorUpd = ((mode >> 3) & 1u) == 1u, alphaUpd = ((mode >> 4) & 1u) == 1u;
  if (colorUpd) {
    if (fmt == 1u) {
      uint a6 = alphaUpd ? uint(r.a >> 2) : (old & 63u);
      return a6 | (uint(r.b >> 2) << 6) | (uint(r.g >> 2) << 12) | (uint(r.r >> 2) << 18);
    }
    return uint(r.b) | (uint(r.g) << 8) | (uint(r.r) << 16);
  }
  if (alphaUpd && fmt == 1u) return (old & ~63u) | uint(r.a >> 2);
  return old;
}

// ---------------------------------------------------------------------------------------------------------------------
// one pixel of one triangle (spec 8 to 11)

void shade(int x, int y, inout uint col, inout uint dep) {
  uint stIdx = gState;
  sbase = stIdx * STATE_WORDS;
  uint b0 = S(0x00u);
  int nStages = bfi(b0, 10, 4) + 1;
  int numCols = bfi(b0, 4, 3);
  uint zmode = S(0x40u);
  bool zEnable = (zmode & 1u) == 1u, zUpdate = ((zmode >> 4) & 1u) == 1u;
  uint zFunc = (zmode >> 1) & 7u;
  uint b43 = S(0x43u);
  uint pixFmt = b43 & 7u;
  bool zEarly = ((b43 >> 6) & 1u) == 1u && zEnable;

  uint dx = f_add(gOx, f_from_int(x - gXi)), dy = f_add(gOy, f_from_int(y - gYi));
  uint zf = planeAt(1u, dx, dy);
  int z = f_trunc(f_gt(zf, 0x4b7fffffu) ? 0x4b7fffffu : (f_lt(zf, 0u) ? 0u : zf)) & 0xffffff;
  int stored = int(dep);
  uint newDep = dep;
  if (zEarly) {
    if (!cmpOp(zFunc, z, stored)) return;
    if (zUpdate) newDep = uint(z);
  }
  // raster colors
  for (int i = 0; i < 8; i++) gRasCh[i] = 0;
  for (int i = 0; i < numCols * 4; i++) {
    int tv = f_trunc(planeAt(uint(2 + i), dx, dy));
    int v16 = tv & 0xffff;
    gRasCh[i] = (v16 & ~(v16 >> 8)) & 0xff;
  }
  loadColorRegs();
  for (int i = 0; i < 4; i++) { gLastTex[i] = 0; gTex[i] = 0; }
  int prevS = 0, prevT = 0;
  for (int si = 0; si < nStages; si++) {
    uint ord = S(0x28u + uint(si >> 1)) >> ((si & 1) == 1 ? 12u : 0u);
    uint texmap = bf(ord, 0, 3), texcoord = bf(ord, 3, 3), rasSel = bf(ord, 7, 3);
    bool texEn = bf(ord, 6, 1) == 1u;
    uint cc = S(0xc0u + uint(2 * si)), ac = S(0xc1u + uint(2 * si));
    uint kw = S(0xf6u + uint(si >> 1));
    int kcs = ((si & 1) == 1) ? bfi(kw, 14, 5) : bfi(kw, 4, 5);
    int kas = ((si & 1) == 1) ? bfi(kw, 19, 5) : bfi(kw, 9, 5);
    uint rsw = bf(ac, 0, 2), tsw = bf(ac, 2, 2);
    int bump = 0;
#ifdef INDIRECT
    // with the indirect unit three coordinate computations and two samples go through one copy of each, by loops with run-time bounds:
    // coordinates job 0 the level of detail of the indirect stage (the low two bits of its map and coordinate numbers, a quirk of the
    // reference), job 1 its coordinates, job 2 the stage's own; sample job 0 the indirect sample, job 1 the stage's
    uint indWord = bfi(b0, 16, 3) > 0 ? S(0x10u + uint(si)) : 0u;
    bool hasInd = indWord != 0u;
    int ibt = bfi(indWord, 0, 2);
    uint iref = S(0x27u);
    int imap = bfi(iref, 6 * ibt, 3), icoord = bfi(iref, 6 * ibt + 3, 3);
    uint ssw = S(0x25u + uint(ibt >> 1));
    int iss = bfi(ssw, (ibt & 1) * 8, 4), its = bfi(ssw, (ibt & 1) * 8 + 4, 4);
    int jS[3], jT[3], jL[3];
    bool jLin[3];
    for (int j = 0; j < uK.z; j++) {
      jS[j] = 0; jT[j] = 0; jL[j] = 0; jLin[j] = false;
      if (j == 2 ? texEn : hasInd) {
        uint mapj = j == 0 ? uint(imap & 3) : (j == 1 ? uint(imap) : texmap);
        uint genj = j == 0 ? uint(icoord & 3) : (j == 1 ? uint(icoord) : texcoord);
        stageCoords(loadMap(mapj), genj, x, y, jS[j], jT[j], jL[j], jLin[j]);
      }
    }
    int mS = jS[2], mT = jT[2];
    for (int k = 0; k < uK.w; k++) {
      if (k == 0 ? hasInd : texEn) {
        MapI mps = loadMap(k == 0 ? uint(imap) : texmap);
        ivec4 r = sampleMap(mps, k == 0 ? jS[1] >> iss : mS, k == 0 ? jT[1] >> its : mT, k == 0 ? jL[0] : jL[2], k == 0 ? jLin[0] : jLin[2]);
        if (k == 0) {
          int oS, oT;
          indirectOffsets(indWord, r, jS[2], jT[2], prevS, prevT, oS, oT, bump);
          mS = oS; mT = oT;
        } else {
          prevS = mS; prevT = mT;
          int tv[4] = int[4](r.r, r.g, r.b, r.a);
          swz(tsw, tv, gTex);
          for (int i = 0; i < 4; i++) gLastTex[i] = gTex[i];
        }
      }
    }
#else
    if (texEn) {
      MapI mp = loadMap(texmap);
      int cS, cT, lod;
      bool lin;
      stageCoords(mp, texcoord, x, y, cS, cT, lod, lin);
      prevS = cS; prevT = cT;
      ivec4 tx = sampleMap(mp, cS, cT, lod, lin);
      int tv[4] = int[4](tx.r, tx.g, tx.b, tx.a);
      swz(tsw, tv, gTex);
      for (int i = 0; i < 4; i++) gLastTex[i] = gTex[i];
    }
#endif
    for (int i = 0; i < 4; i++) gRas[i] = 0;
    if (rasSel == 0u || rasSel == 1u) {
      int rv[4] = int[4](gRasCh[int(rasSel) * 4], gRasCh[int(rasSel) * 4 + 1], gRasCh[int(rasSel) * 4 + 2], gRasCh[int(rasSel) * 4 + 3]);
      swz(rsw, rv, gRas);
    } else if (rasSel == 5u || rasSel == 6u) {
      int bv = rasSel == 5u ? bump : (bump | (bump >> 5));
      gRas[0] = bv; gRas[1] = bv; gRas[2] = bv; gRas[3] = bv;
    }
    ivec3 ca = csel(bfi(cc, 12, 4)), cb = csel(bfi(cc, 8, 4)), ccc = csel(bfi(cc, 4, 4)), cd = csel(bfi(cc, 0, 4));
    if (bfi(cc, 12, 4) == 14) ca = konstColor(kcs);
    if (bfi(cc, 8, 4) == 14) cb = konstColor(kcs);
    if (bfi(cc, 4, 4) == 14) ccc = konstColor(kcs);
    if (bfi(cc, 0, 4) == 14) cd = konstColor(kcs);
    int cbias = bfi(cc, 16, 2), cop = bfi(cc, 18, 1), cclamp = bfi(cc, 19, 1), cscale = bfi(cc, 20, 2), cdest = bfi(cc, 22, 2);
    int aa = asel(bfi(ac, 13, 3), kas), ab = asel(bfi(ac, 10, 3), kas), ac_ = asel(bfi(ac, 7, 3), kas), ad = asel(bfi(ac, 4, 3), kas);
    int abias = bfi(ac, 16, 2), aop = bfi(ac, 18, 1), aclamp = bfi(ac, 19, 1), ascale = bfi(ac, 20, 2), adest = bfi(ac, 22, 2);
    ivec3 rcv;
    ivec3 dv = ivec3(sx11(uint(cd.x) & 0x7ffu), sx11(uint(cd.y) & 0x7ffu), sx11(uint(cd.z) & 0x7ffu));
    if (cbias == 3) rcv = clampTev3(dv + ivec3(compare3((cscale << 1) | cop, ca & 255, cb & 255)) * (ccc & 255), cclamp);
    else rcv = tevColor3(ca & 255, cb & 255, ccc & 255, dv, cbias, cop, cscale, cclamp);
    int rc[3] = int[3](rcv.x, rcv.y, rcv.z);
    int ra;
    if (abias == 3) {
      int mode = (ascale << 1) | aop;
      bool cond;
      if (mode >= 6) cond = mode == 6 ? ((aa & 255) > (ab & 255)) : ((aa & 255) == (ab & 255));
      else cond = compareCond(mode, ca & 255, cb & 255, 0);
      ra = clampTev(sx11(uint(ad) & 0x7ffu) + (cond ? (ac_ & 255) : 0), aclamp);
    } else ra = tevRegular(true, aa & 255, ab & 255, ac_ & 255, sx11(uint(ad) & 0x7ffu), abias, aop, ascale, aclamp);
    gRegs[cdest * 4] = rc[0]; gRegs[cdest * 4 + 1] = rc[1]; gRegs[cdest * 4 + 2] = rc[2];
    gRegs[adest * 4 + 3] = ra;
  }
  int lastC = bfi(S(0xc0u + uint(2 * (nStages - 1))), 22, 2), lastA = bfi(S(0xc1u + uint(2 * (nStages - 1))), 22, 2);
  ivec4 src = ivec4(gRegs[lastC * 4] & 255, gRegs[lastC * 4 + 1] & 255, gRegs[lastC * 4 + 2] & 255, gRegs[lastA * 4 + 3] & 255);
  // alpha test
  uint at = S(0xf3u);
  bool a0 = cmpOp(bf(at, 16, 3), src.a, bfi(at, 0, 8)), a1 = cmpOp(bf(at, 19, 3), src.a, bfi(at, 8, 8));
  uint lg = bf(at, 22, 2);
  bool pass = lg == 0u ? (a0 && a1) : (lg == 1u ? (a0 || a1) : (lg == 2u ? (a0 != a1) : (a0 == a1)));
  if (!pass) { dep = newDep; return; }
  // depth texture
  uint dtOp = bf(S(0xf5u), 2, 2);
  if (dtOp != 0u) {
    uint sel = bf(S(0xf5u), 0, 2);
    int X = sel == 0u ? gLastTex[3] : (sel == 1u ? ((gLastTex[3] << 8) | gLastTex[0]) : ((gLastTex[0] << 16) | (gLastTex[1] << 8) | gLastTex[2]));
    z = (int(S(0xf4u)) + X + (dtOp == 1u ? z : 0)) & 0xffffff;
  }
  uint fsel = bf(S(0xf1u), 21, 3);
  if (fsel != 0u) fogPixel(fsel, z, x, src);
  if (!zEarly && zEnable) {
    if (!cmpOp(zFunc, z, stored)) { dep = newDep; return; }
    if (zUpdate) newDep = uint(z);
  }
  ivec4 dst = readColor(col, pixFmt);
  ivec4 res = blendPixel(S(0x41u), S(0x42u), src, dst);
  col = writeColor(col, pixFmt, S(0x41u), res);
#ifdef DEBUG_REGS
  // a test aid: the color word carries two of the TEV registers' components instead of the pixel
  col = (uint(gRegs[DEBUG_A] & 0xfff) << 12) | uint(gRegs[DEBUG_B] & 0xfff);
#endif
  dep = newDep;
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  uvec2 e = texelFetch(uEfb, p, 0).rg;
  uint col = e.x, dep = e.y;
  int tile = (p.x >> ${D}) + (p.y >> ${D}) * uTilesX;
  uvec2 ti = texelFetch(uTiles, ivec2(tile & 1023, tile >> 10), 0).rg;
  int sub = ((p.x >> 3) & 1) + (((p.y >> 3) & 1) << 1);
  for (uint i = 0u; i < ti.y; i++) {
    uint ent = wd(uList, ti.x + i);
    if (((ent >> uint(24 + sub)) & 1u) == 0u) continue;
    uint tri = ent & 0xffffffu;
    uvec4 h0 = td(tri);
    int bx = p.x & ~1, by = p.y & ~1;
    if (bx < int(h0.x) || bx >= int(h0.y) || by < int(h0.z) || by >= int(h0.w)) continue;
    uvec4 h1 = td(tri + 1u), h2 = td(tri + 2u), h3 = td(tri + 3u);
    int DX12 = int(h1.x), DY12 = int(h1.y), DX23 = int(h1.z), DY23 = int(h1.w), DX31 = int(h2.x), DY31 = int(h2.y);
    int C1 = int(h2.z), C2 = int(h2.w), C3 = int(h3.x);
    if (C1 + DX12 * (16 * p.y) - DY12 * (16 * p.x) <= 0) continue;
    if (C2 + DX23 * (16 * p.y) - DY23 * (16 * p.x) <= 0) continue;
    if (C3 + DX31 * (16 * p.y) - DY31 * (16 * p.x) <= 0) continue;
    uvec4 h4 = td(tri + 4u);
    gTri = tri;
    gXi = int(h3.y); gYi = int(h3.z);
    gOx = h3.w; gOy = h4.x;
    gState = h4.y;
    gNp = h4.z;
    shade(p.x, p.y, col, dep);
  }
  o = uvec4(col, dep, 0u, 0u);
}
`,i=e.floatOps!=="dual"?t:t.replace("  uint col = e.x, dep = e.y;","  uint col = e.x, dep = e.y & 0xffffffu;").replace(`  o = uvec4(col, dep, 0u, 0u);
}`,`  o = uvec4(col, gFloatMismatch ? (dep | 0x80000000u) : dep, 0u, 0u);
}`);if(e.floatOps==="dual"&&!i.includes("gFloatMismatch ? (dep"))throw new Error("pixelShader: no place for the dual check");return e.mutate?e.mutate(i):i}G();function zi(){const e=200n,t=1n<<e;let i=0n;for(let r=1n;r<220n;r++)i+=t/(r<<r);const n=r=>{let f=t,o=t;for(let a=1n;f!==0n;a++)f=f*r/t/a,o+=a%2n===1n?-f:f;return o},s=r=>{const f=r+(1n<<e-65n)>>e-64n,o=f>=1n<<64n?(1n<<64n)-1n:f;return[Number(o>>48n&0xffffn),Number(o>>32n&0xffffn),Number(o>>16n&0xffffn),Number(o&0xffffn)]},u=new Uint32Array(32);for(let r=1;r<=6;r++)u.set(s(n(i>>BigInt(r))),(r-1)*4);return u.set(s(i),24),u[28]=8,u}const Ii=`#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uEfb;
uniform int uDest, uStride, uFmt, uBlocksX, uBlocksY, uKw, uKh, uBytes, uHalf, uX0, uY0, uPixFmt;
out uint o;

int w6(int v) { return (v << 2) | (v >> 4); }

ivec4 pixel(int p) {
  if (p < 0 || p >= 640 * 528) return ivec4(0);
  if (uPixFmt == 3) {
    uint z = texelFetch(uEfb, ivec2(p % 640, p / 640), 0).g;
    return ivec4(int((z >> 16) & 255u), int((z >> 8) & 255u), int(z & 255u), 255);
  }
  uint w = texelFetch(uEfb, ivec2(p % 640, p / 640), 0).r;
  if (uPixFmt == 1) return ivec4(w6(int((w >> 18) & 63u)), w6(int((w >> 12) & 63u)), w6(int((w >> 6) & 63u)), w6(int(w & 63u)));
  return ivec4(int((w >> 16) & 255u), int((w >> 8) & 255u), int(w & 255u), 255);
}

// texel (X, Y) of the output: r, g, b, a
ivec4 texel(int X, int Y) {
  int k = uHalf == 1 ? 2 : 1;
  int p = (uY0 + Y * k) * 640 + uX0 + X * k;
  if (uHalf == 0) return pixel(p);
  ivec4 s = pixel(p) + pixel(p + 1) + pixel(p + 640) + pixel(p + 641);
  if (uPixFmt == 1) return min(ivec4(255), s + (s >> 6));
  return s >> 2;
}

int lum(ivec4 c) { return (4096 + 66 * c.r + 129 * c.g + 25 * c.b) >> 8; }

// the byte of one texel for the 8-bit and the 16-bit (hi / lo) formats
int texByte(int fmt, ivec4 c, int part) {
  if (fmt == 1) return lum(c);
  if (fmt == 2) return 0xf0 | (lum(c) >> 4);
  if (fmt == 3) return part == 0 ? 255 : lum(c);
  if (fmt == 4) { int v = ((c.r << 8) & 0xf800) | ((c.g << 3) & 0x07e0) | ((c.b >> 3) & 0x001e); return part == 0 ? v >> 8 : v & 255; }
  if (fmt == 5) { int v = 0x8000 | ((c.r << 7) & 0x7c00) | ((c.g << 2) & 0x03e0) | ((c.b >> 3) & 0x001e); return part == 0 ? v >> 8 : v & 255; }
  if (fmt == 8) return 0xf0 | (c.r >> 4);
  if (fmt == 9) return part == 0 ? 255 : c.r;
  if (fmt == 10) return 255;
  if (fmt == 11) return c.r;
  if (fmt == 12) return c.g;
  if (fmt == 13) return c.b;
  if (fmt == 14) return part == 0 ? c.g : c.r;
  if (fmt == 16) return uHalf == 1 ? (part == 0 ? c.r : c.g) : (part == 0 ? c.g : c.r);
  if (fmt == 17) return uHalf == 1 ? (part == 0 ? c.g : c.b) : (part == 0 ? c.b : c.g);
  return part == 0 ? c.b : c.g;
}

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
  int a = fc.y * ${y} + fc.x;
  int rel = a - uDest;
  if (rel < 0) discard;
  int r = uStride > 0 ? rel / uStride : 0;
  int inRow = rel - r * uStride;
  if (r >= uBlocksY || inRow >= uBlocksX * uBytes) discard;
  int blk = inRow / uBytes;
  int off = inRow - blk * uBytes;
  int bw = 1 << uKw, bh = 1 << uKh;
  int fmt = uFmt;
  int res = 0;
  if (fmt == 6) {
    int q = (off & 31) >> 1;
    ivec4 c = texel(blk * bw + (q & 3), r * bh + (q >> 2));
    // a half-size depth copy writes (255, z0), (z1, z2)
    if (uPixFmt == 3 && uHalf == 1) c = c.bgra;
    int hiPart = off >= 32 ? 1 : 0;
    int odd = off & 1;
    res = hiPart == 0 ? (odd == 0 ? c.a : c.r) : (odd == 0 ? c.g : c.b);
  } else if (fmt == 0 || fmt == 7) {
    int j = off >> 2, i = (off & 3) * 2;
    ivec4 ca = texel(blk * bw + i, r * bh + j), cb = texel(blk * bw + i + 1, r * bh + j);
    int ya = fmt == 0 ? lum(ca) : ca.r, yb = fmt == 0 ? lum(cb) : cb.r;
    res = (ya & 0xf0) | (yb >> 4);
  } else if (bw == 4) {
    int q = off >> 1;
    ivec4 c = texel(blk * bw + (q & 3), r * bh + (q >> 2));
    res = texByte(fmt, c, off & 1);
  } else {
    int j = off >> 3, i = off & 7;
    ivec4 c = texel(blk * bw + i, r * bh + j);
    res = texByte(fmt, c, 0);
  }
  o = uint(res & 255);
}
`,Di=`#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uEfb;
uniform ivec4 uRect;      // x0, y0, x1, y1 inclusive
uniform uint uColor;      // the color word to write if color update
uniform uint uAlpha6;     // 6-bit alpha (RGBA6)
uniform uint uDepth;
uniform int uColorUpd, uAlphaUpd, uZUpd, uPixFmt;
out uvec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  uvec2 e = texelFetch(uEfb, p, 0).rg;
  if (p.x >= uRect.x && p.x <= uRect.z && p.y >= uRect.y && p.y <= uRect.w) {
    if (uColorUpd == 1) {
      if (uPixFmt == 1) e.x = (uAlphaUpd == 1 ? uAlpha6 : (e.x & 63u)) | (uColor & ~63u);
      else e.x = uColor;
    } else if (uAlphaUpd == 1 && uPixFmt == 1) e.x = (e.x & ~63u) | uAlpha6;
    if (uZUpd == 1) e.y = uDepth;
  }
  o = uvec4(e, 0u, 0u);
}
`,Ci=`#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uEfb;
uniform int uPixFmt;
out vec4 o;
int w6(int v) { return (v << 2) | (v >> 4); }
void main() {
  uint w = texelFetch(uEfb, ivec2(gl_FragCoord.xy), 0).r;
  ivec4 c;
  if (uPixFmt == 1) c = ivec4(w6(int((w >> 18) & 63u)), w6(int((w >> 12) & 63u)), w6(int((w >> 6) & 63u)), w6(int(w & 63u)));
  else c = ivec4(int((w >> 16) & 255u), int((w >> 8) & 255u), int(w & 255u), 255);
  o = vec4(c) / 255.0;
}
`,Pi=`#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uEfb;
uniform int uPixFmt;
uniform vec2 uSize;      // the viewport's width and height in pixels
out vec4 o;
int w6(int v) { return (v << 2) | (v >> 4); }
void main() {
  ivec2 p = ivec2(floor(vec2(gl_FragCoord.x, uSize.y - gl_FragCoord.y) * vec2(640.0, 480.0) / uSize));
  p = clamp(p, ivec2(0), ivec2(639, 479));
  uint w = texelFetch(uEfb, p, 0).r;
  ivec3 c;
  if (uPixFmt == 1) c = ivec3(w6(int((w >> 18) & 63u)), w6(int((w >> 12) & 63u)), w6(int((w >> 6) & 63u)));
  else c = ivec3(int((w >> 16) & 255u), int((w >> 8) & 255u), int(w & 255u));
  o = vec4(vec3(c) / 255.0, 1.0);
}
`;function U(e,t,i){const n=(f,o)=>{const a=e.createShader(f);return e.shaderSource(a,o),e.compileShader(a),a},s=n(e.VERTEX_SHADER,t),u=n(e.FRAGMENT_SHADER,i),r=e.createProgram();return e.attachShader(r,s),e.attachShader(r,u),e.linkProgram(r),{program:r,vs:s,fs:u,fsSource:i}}function ht(e,t){const i=e.getExtension("KHR_parallel_shader_compile");return i?e.getProgramParameter(t.program,i.COMPLETION_STATUS_KHR)===!0:!0}function Wt(e,t){for(const[i,n]of[[t.vs,""],[t.fs,t.fsSource]])if(!e.getShaderParameter(i,e.COMPILE_STATUS)){const s=e.getShaderInfoLog(i)??"",u=n.split(`
`),r=/ERROR: \d+:(\d+)/.exec(s),f=r?Number(r[1]):0;throw new Error(`shader compile failed: ${s}
${u.slice(Math.max(0,f-4),f+3).join(`
`)}`)}if(!e.getProgramParameter(t.program,e.LINK_STATUS))throw new Error("program link failed: "+e.getProgramInfoLog(t.program));return t.program}const M=`#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`,Vt=e=>({R8UI:{internal:e.R8UI,format:e.RED_INTEGER,type:e.UNSIGNED_BYTE},R32UI:{internal:e.R32UI,format:e.RED_INTEGER,type:e.UNSIGNED_INT},RG32UI:{internal:e.RG32UI,format:e.RG_INTEGER,type:e.UNSIGNED_INT},RGBA32UI:{internal:e.RGBA32UI,format:e.RGBA_INTEGER,type:e.UNSIGNED_INT}});function I(e,t,i,n){const s=e.createTexture();return e.bindTexture(e.TEXTURE_2D,s),e.texStorage2D(e.TEXTURE_2D,1,Vt(e)[t].internal,i,n),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MIN_FILTER,e.NEAREST),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MAG_FILTER,e.NEAREST),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE),{tex:s,w:i,h:n,fmt:t}}function B(e,t,i,n,s,u,r){const f=Vt(e)[t.fmt];e.bindTexture(e.TEXTURE_2D,t.tex),e.pixelStorei(e.UNPACK_ALIGNMENT,1),e.texSubImage2D(e.TEXTURE_2D,0,i,n,s,u,f.format,f.type,r)}const Bt=0,rt=1,Kt=2,bt=3,Ni=4,Oi=10,Ut=16,O=20,Ht=0,Jt=1,Yt=2,xt=3,$t=4,Zt=5,Qt=6,ti=7,ii=8,ei=9,wt=10,gt=11,X=12,ot=15,ni=16,si=17,Li={maxJobs:512,xfSlots:512,inWords:1<<23,outWords:1<<24};class at{constructor(t,i){this.sab=t,this.cfg=i;const n=at.layout(i);this.ctl=new Int32Array(t,0,Ut),this.jobs=new Int32Array(t,n.jobsAt*4,i.maxJobs*O),this.jobsF=new Float32Array(t,n.jobsAt*4,i.maxJobs*O),this.xf=new Uint32Array(t,n.xfAt*4,i.xfSlots*q),this.inU=new Uint32Array(t,n.inAt*4,i.inWords),this.inI=new Int32Array(t,n.inAt*4,i.inWords),this.inF=new Float32Array(t,n.inAt*4,i.inWords),this.inB=new Uint8Array(t,n.inAt*4,i.inWords*4),this.out=new Uint32Array(t,n.outAt*4,i.outWords)}ctl;jobs;jobsF;xf;inU;inI;inF;inB;out;static layout(t){const i=Ut,n=i+t.maxJobs*O,s=n+t.xfSlots*q,u=s+t.inWords;return{jobsAt:i,xfAt:n,inAt:s,outAt:u,words:u+t.outWords}}}function ji(e){const t=(e.posMtx?1:0)|e.pos.kind<<1|e.nrm.kind<<3|(e.nbt?32:0)|e.col[0].kind<<6|e.col[1].kind<<8;let i=0;for(let n=0;n<8;n++)i|=e.tex[n].kind<<2*n|(e.texMtx[n]?1:0)<<16+n;return[t,i]}const Mt=new Map;function Gi(e,t){const i=e+","+t;let n=Mt.get(i);if(!n){const s=u=>({kind:u});n={posMtx:(e&1)!==0,pos:s(e>>1&3),nrm:s(e>>3&3),nbt:(e&32)!==0,col:[s(e>>6&3),s(e>>8&3)],tex:Array.from({length:8},(u,r)=>s(t>>2*r&3)),texMtx:Array.from({length:8},(u,r)=>(t>>16+r&1)!==0)},Mt.set(i,n)}return n}function vt(e,t,i,n){const s={pos:n,pmtx:n+3*i,nrm:-1,nrm1:-1,nrm2:-1,col:[-1,-1],tex:[],tmtx:[],end:0};let u=n+4*i;e>>3&3&&(s.nrm=u,u+=3*i),e&32&&(s.nrm1=u,u+=3*i,s.nrm2=u,u+=3*i);for(let r=0;r<2;r++)e>>6+2*r&3&&(s.col[r]=u,u+=i);for(let r=0;r<8;r++)t>>2*r&3?(s.tex.push(u),u+=2*i):s.tex.push(-1);for(let r=0;r<8;r++)t>>16+r&1?(s.tmtx.push(u),u+=i):s.tmtx.push(-1);return s.end=u,s}const H=new Float32Array(0),zt=new Uint8Array(0);function ri(e,t,i,n){const s=e.jobs,u=n%e.cfg.maxJobs*O,r=s[u+Ht],f=s[u+Jt],o=s[u+$t],a=vt(r,f,o,s[u+Zt]),c=e.inF,h={n:o,layout:Gi(r,f),pos:c.subarray(a.pos,a.pos+3*o),pmtx:c.subarray(a.pmtx,a.pmtx+o),nrm:a.nrm>=0?c.subarray(a.nrm,a.nrm+3*o):H,nrm1:a.nrm1>=0?c.subarray(a.nrm1,a.nrm1+3*o):H,nrm2:a.nrm2>=0?c.subarray(a.nrm2,a.nrm2+3*o):H,col:[a.col[0]>=0?e.inB.subarray(a.col[0]*4,a.col[0]*4+4*o):zt,a.col[1]>=0?e.inB.subarray(a.col[1]*4,a.col[1]*4+4*o):zt],tex:a.tex.map(l=>l>=0?c.subarray(l,l+2*o):H),tmtx:a.tmtx.map(l=>l>=0?c.subarray(l,l+o):H)},p=s[u+ti],x=s[u+Qt],g=[];for(let l=0;l<p;l++)g.push({type:e.inI[x+3*l],base:e.inI[x+3*l+1],count:e.inI[x+3*l+2]});const d=s[u+Yt],w={bp:e.inU.subarray(s[u+xt],s[u+xt]+256),xf:e.xf.subarray(d*q,(d+1)*q)};i.n=0,ut.wrote=!1,t.run(w,h,g,s[u+ii],i)}const nt=new Float32Array(3);function Xi(e,t,i,n,s){ri(e,t,i,s);const u=e.jobs,r=s%e.cfg.maxJobs*O,f=i.n;n.n=0;const o=dt(i.ints,i.words,f,0,n),a=Atomics.add(e.ctl,bt,f+n.n);a+f+n.n>e.cfg.outWords?u[r+ot]=1:(e.out.set(i.words.subarray(0,f),a),e.out.set(new Uint32Array(n.a.buffer,0,n.n),a+f),u[r+ei]=a,u[r+wt]=f,u[r+ni]=n.n,u[r+si]=o,u[r+ot]=0),u[r+gt]=ut.wrote?1:0,ut.wrote&&(Pt(nt),e.jobsF[r+X]=nt[0],e.jobsF[r+X+1]=nt[1],e.jobsF[r+X+2]=nt[2]),Atomics.add(e.ctl,Kt,1)}function qi(e){const t=Atomics.load(e.ctl,rt),i=Atomics.load(e.ctl,Bt);return i<t&&Atomics.compareExchange(e.ctl,Bt,i,i+1)===i?i:-1}class It{sh;workers=[];ctx=new Ot;lq=new pt;le=new jt;base=0;tail=0;inUsed=0;xfUsed=0;xfVersion=-1;xfSlot=-1;zPlane=null;zFresh=!1;stats={mainJobs:0,jobs:0,waitMs:0,mergeMs:0};static available(){return typeof SharedArrayBuffer=="function"&&typeof Atomics<"u"&&globalThis.crossOriginIsolated!==!1}constructor(t,i,n=Li){const s=new SharedArrayBuffer(at.layout(n).words*4);this.sh=new at(s,n);for(let u=0;u<t;u++){const r=i();r.onerror=f=>console.error("front worker: "+f.message),r.postMessage({sab:s,cfg:n}),this.workers.push(r)}}dispose(){Atomics.store(this.sh.ctl,Ni,1),Atomics.notify(this.sh.ctl,rt);for(const t of this.workers)t.terminate();this.workers=[]}get workerJobs(){return Atomics.load(this.sh.ctl,8)}get workersStarted(){return Atomics.load(this.sh.ctl,9)}get busy(){return this.tail>this.base}submit(t,i,n,s,u){const r=this.sh,f=i.n,[o,a]=ji(i.layout),c=256+3*n.length+vt(o,a,f,0).end;(this.tail-this.base>=r.cfg.maxJobs||this.inUsed+c>r.cfg.inWords||this.xfUsed>=r.cfg.xfSlots||Atomics.load(r.ctl,bt)>r.cfg.outWords/2)&&u(),(this.xfSlot<0||this.xfVersion!==t.xfVersion)&&(this.xfSlot=this.xfUsed++,r.xf.set(t.xf.subarray(0,q),this.xfSlot*q),this.xfVersion=t.xfVersion);let h=this.inUsed;const p=h;r.inU.set(t.bp.subarray(0,256),h),h+=256;const x=h;for(let m=0;m<n.length;m++)r.inI[h++]=n[m].type,r.inI[h++]=n[m].base,r.inI[h++]=n[m].count;const g=h,d=vt(o,a,f,g),w=r.inF;w.set(i.pos.subarray(0,3*f),d.pos),w.set(i.pmtx.subarray(0,f),d.pmtx),d.nrm>=0&&w.set(i.nrm.subarray(0,3*f),d.nrm),d.nrm1>=0&&(w.set(i.nrm1.subarray(0,3*f),d.nrm1),w.set(i.nrm2.subarray(0,3*f),d.nrm2));for(let m=0;m<2;m++)d.col[m]>=0&&r.inB.set(i.col[m].subarray(0,4*f),d.col[m]*4);for(let m=0;m<8;m++)d.tex[m]>=0&&w.set(i.tex[m].subarray(0,2*f),d.tex[m]),d.tmtx[m]>=0&&w.set(i.tmtx[m].subarray(0,f),d.tmtx[m]);this.inUsed=d.end;const l=r.jobs,b=this.tail%r.cfg.maxJobs*O;l[b+Ht]=o,l[b+Jt]=a,l[b+Yt]=this.xfSlot,l[b+xt]=p,l[b+$t]=f,l[b+Zt]=g,l[b+Qt]=x,l[b+ti]=n.length,l[b+ii]=s,l[b+ot]=0,l[b+wt]=0,l[b+gt]=0,this.tail++,Atomics.store(r.ctl,rt,this.tail),Atomics.load(r.ctl,Oi)>0&&Atomics.notify(r.ctl,rt,1)}drain(t){const i=this.sh;if(this.tail===this.base)return;const n=this.tail,s=performance.now();for(this.stats.jobs+=n-this.base;;){const r=qi(i);if(r>=0){Xi(i,this.ctx,this.lq,this.le,r),this.stats.mainJobs++;continue}if(Atomics.load(i.ctl,Kt)>=n)break}const u=performance.now();this.stats.waitMs+=u-s;for(let r=this.base;r<n;r++){const f=r%i.cfg.maxJobs*O;if(i.jobs[f+ot])ri(i,this.ctx,this.lq,r),t(this.lq.words.subarray(0,this.lq.n),null,0),ut.wrote&&(this.zPlane=this.zPlane??new Float32Array(3),Pt(this.zPlane),this.zFresh=!0);else{const o=i.jobs[f+ei],a=i.jobs[f+wt];t(i.out.subarray(o,o+a),new Int32Array(i.out.buffer,i.out.byteOffset+(o+a)*4,i.jobs[f+ni]),i.jobs[f+si]),i.jobs[f+gt]&&(this.zPlane=this.zPlane??new Float32Array(3),this.zPlane[0]=i.jobsF[f+X],this.zPlane[1]=i.jobsF[f+X+1],this.zPlane[2]=i.jobsF[f+X+2],this.zFresh=!0)}}this.stats.mergeMs+=performance.now()-u,this.base=n,this.inUsed=0,this.xfUsed=0,this.xfSlot=-1,Atomics.store(i.ctl,bt,0)}}class Wi{out=[];u32(t){this.out.push(t>>>24&255,t>>>16&255,t>>>8&255,t&255)}u16(t){this.out.push(t>>>8&255,t&255)}bp(t,i){return this.out.push(97),this.u32((t<<24|i&16777215)>>>0),this}bpMasked(t,i,n){return this.bp(254,n),this.bp(t,i)}cp(t,i){return this.out.push(8,t&255),this.u32(i>>>0),this}xf(t,i){this.out.push(16),this.u32((i.length-1<<16|t)>>>0);for(let n=0;n<i.length;n++)this.u32(i[n]>>>0);return this}xfFloats(t,i){const n=new DataView(new ArrayBuffer(4));return this.xf(t,Array.from(i,s=>(n.setFloat32(0,s),n.getUint32(0))))}draw(t,i,n,s){this.out.push(128|t<<3|i),this.u16(n);for(let u=0;u<s.length;u++)this.out.push(s[u]&255);return this}bytes(){return Uint8Array.from(this.out)}}function Vi(e=new Wi){const t=[1,0,0,0,0,1,0,0,0,0,1,0];return e.xfFloats(0,t),e.xfFloats(4128,[2/640,-1,-2/480,1,-1/1e3,-1]),e.xf(4134,[1]),e.xfFloats(4122,[320,-240,16777215,320+342,240+342,16777215]),e.cp(80,1<<9|1<<13),e.cp(96,0),e.cp(112,1|4<<1|5<<14|1<<13),e.xf(4105,[1]),e.xf(4110,[1]),e.xf(4112,[1]),e.bp(0,1<<4|0<<10),e.bp(40,0),e.bp(192,10<<0|1<<19),e.bp(193,5<<4|1<<19),e.bp(64,0),e.bp(65,1<<3|1<<4),e.bp(89,171|171<<10),e.bp(32,342<<12|342),e.bp(33,342+639<<12|342+479),e.bp(67,0),e.bp(246,0|1<<2).bp(247,2|3<<2),e.bp(243,7<<16|7<<19),e}const ui=2143289344,P=2139095040;function Ki(e){return 31-Math.clz32(e)}function fi(e,t,i,n,s){const u=n&(s|i&1);i+=u,i===16777216&&(i=8388608,t+=1);let r=(Math.min(Math.max(t,0),255)<<23|i&8388607)>>>0;return t>=255&&(r=P),t<=0&&(r=0),(e<<31|r)>>>0}function Dt(e,t){e>>>=0,t>>>=0;const i=(e&2147483647)>>>0,n=(t&2147483647)>>>0,s=i<n,u=s?t:e,r=s?e:t,f=s?n:i,o=s?i:n,a=u>>>31,c=r>>>31,h=f>>>23,p=o>>>23,x=h===0?0:(f&8388607|8388608)<<3>>>0;let g=p===0?0:(o&8388607|8388608)<<3>>>0;const d=Math.min(h-p,31),w=(g&(1<<d>>>0)-1)>>>0;g=(g>>>d|(w!==0?1:0))>>>0;const l=(a===c?x+g:x-g)>>>0,b=l>=134217728,m=b||l===0?0:Math.max(0,26-Ki(l)),A=(b?l>>>1|l&1:l<<m)>>>0,_=h+(b?1:0)-m;let k=fi(a,_,A>>>3,A>>>2&1,(A&3)!==0?1:0);l===0&&(k=0),h===0&&(k=(e&t&2147483648)>>>0);const S=f>P||o===P&&a!==c?ui:u;return h===255?S:k}function Ct(e,t){e>>>=0,t>>>=0;const i=(e^t)>>>31,n=(e&2147483647)>>>0,s=(t&2147483647)>>>0,u=n>>>23,r=s>>>23,f=u+r-127,o=n&8388607|8388608,a=s&8388607|8388608,c=o>>>12,h=o&4095,p=a>>>12,x=a&4095,g=h*x,d=h*p+c*x,w=c*p;let l=g+((d&4095)<<12);const b=w+(d>>>12)+Math.floor(l/16777216);l=l%16777216;const m=b>=8388608,A=m?b:b*2|l>>>23,_=m?l>>>23:l>>>22&1,k=m?(l&8388607)!==0?1:0:(l&4194303)!==0?1:0;let S=fi(i,f+(m?1:0),A,_,k);const v=u===0,C=r===0,L=n===P,$=s===P,j=n>P||s>P||L&&C||$&&v;return(v||C)&&(S=i<<31>>>0),(L||$)&&(S=(i<<31|P)>>>0),j?ui:S}function Hi(e=65536,t=1784494561){const i=[0,1,2,1023,4194304,8388607,8388608,8388609,8454143,16777215,16777216,25165824,209715200,209715199,201326592,192937984,192937983,1065353216,1065353217,1073741823,1065353215,1073741824,864026624,855638016,872415232,1266679807,1266679808,1124073472,1056964607,1056964608,1060439283,1602224128,528482304,520093696,536870912,2130706431,2130706432,2139095039,2139095040,2139095041,2143289344,2147483647,16777214,16777216,25165824,427819007,645922816],n=[];for(const o of i)n.push(o>>>0,(o|2147483648)>>>0);const s=new Uint32Array(e*2);let u=0;for(const o of n)for(const a of n){if(u>=e)break;s[2*u]=o,s[2*u+1]=a,u++}let r=t>>>0;const f=()=>(r^=r<<13,r>>>=0,r^=r>>>17,r^=r<<5,r>>>=0,r);for(;u<e;u++){const o=f(),a=f()&7;let c=f();a===0?c=(o&2147483648^f()&2147483648|(253-(o>>>23&255)+f()%5&255)<<23|f()&8388607)>>>0:a===1?c=(o^2147483648)>>>0:a===2?c=(c&2155872255|((o>>>23&255)+f()%50-25&255)<<23)>>>0:a===3&&(c=(o&4286578688|f()&8388607)>>>0),s[2*u]=o,s[2*u+1]=c}return s}const Ji=e=>`#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D uIn;
out uvec4 o;
${e}
void main() {
  uvec2 v = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rg;
  o = uvec4(f_add(v.x, v.y), f_mul(v.x, v.y), f_add(f_mul(v.x, v.y), v.x), f_mul(f_add(v.x, v.y), v.y));
}`;function Yi(e){const n=Hi(65536),s=Wt(e,U(e,M,Ji(Nt("fast")))),u=I(e,"RG32UI",256,256),r=I(e,"RGBA32UI",256,256),f=e.createFramebuffer();try{B(e,u,0,0,256,256,n),e.bindFramebuffer(e.FRAMEBUFFER,f),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,r.tex,0),e.viewport(0,0,256,256),e.disable(e.SCISSOR_TEST),e.useProgram(s),e.activeTexture(e.TEXTURE0),e.bindTexture(e.TEXTURE_2D,u.tex),e.uniform1i(e.getUniformLocation(s,"uIn"),0),e.drawArrays(e.TRIANGLES,0,3);const o=new Uint32Array(256*256*4);e.readPixels(0,0,256,256,e.RGBA_INTEGER,e.UNSIGNED_INT,o);let a=0;for(let c=0;c<256*256;c++){const h=n[2*c],p=n[2*c+1],x=Dt(h,p),g=Ct(h,p);o[4*c]!==x&&a++,o[4*c+1]!==g&&a++,o[4*c+2]!==Dt(g,h)&&a++,o[4*c+3]!==Ct(x,p)&&a++}return a}finally{e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteFramebuffer(f),e.deleteTexture(u.tex),e.deleteTexture(r.tex),e.deleteProgram(s)}}const N=e=>{const t=new DataView(new ArrayBuffer(4));return t.setFloat32(0,e),[t.getUint8(0),t.getUint8(1),t.getUint8(2),t.getUint8(3)]};function oi(){let e=625341585;const t=()=>(e^=e<<13,e>>>=0,e^=e>>>17,e^=e<<5,e>>>=0,e/4294967296),i=new Uint8Array(25165824),n=1048576;for(let r=n;r<n+65536;r++)i[r]=t()*256|0;const s=Vi();s.cp(80,1<<9|1<<13),s.cp(96,1<<0|1<<2),s.cp(112,1|4<<1|5<<14|1<<21|4<<22),s.cp(128,1|4<<1),s.xf(4159,[2]),s.xf(4160,[5<<7]),s.xf(4161,[6<<7|1<<1]);for(let r=0;r<2;r++)s.bp(128+r,(r===0?0:1|2<<2)|1<<4|4<<5),s.bp(132+r,0|32<<8),s.bp(136+r,(r===0?31|31<<10:15|63<<10)|4<<20),s.bp(148+r,n+r*16384>>5),s.bp(48+2*r,r===0?31:15),s.bp(49+2*r,r===0?31:63);s.bp(0,2|1<<4|2<<10),s.bp(40,0|0<<3|1<<6|0<<7|(1|1<<3|1<<6|0<<7)<<12),s.bp(41,0|0<<3|0<<6|0<<7),s.bp(192,15|10<<4|8<<8|15<<12),s.bp(193,7<<4|5<<7|4<<10|7<<13),s.bp(194,0|9<<4|8<<8|15<<12),s.bp(195,0<<4|4<<7|7<<10|7<<13),s.bp(196,0|10<<4|15<<8|15<<12|1<<20),s.bp(197,0<<4|5<<7|7<<10|7<<13),s.bp(243,7<<16|7<<19),s.bp(64,1|6<<1|1<<4),s.bp(65,1|1<<3|1<<4|4<<5|5<<8);for(let r=0;r<14;r++){const f=[],o=8+t()*80,a=8+t()*80,c=4+t()*(r<4?120:40);for(let h=0;h<3;h++){const p=r%5===4&&h===2?.01:1,x=o+(t()*2-1)*c*p,g=a+(t()*2-1)*c,d=-(1+t()*(r%3===0?1e6:900));f.push(...N(x),...N(g),...N(d),t()*256|0,t()*256|0,t()*256|0,t()*256|0);const w=r%4===1?1e5:r%4===2?1e-30:1;f.push(...N((t()*2-.5)*w),...N((t()*2-.5)*w),...N(t()*3-1),...N(r%6===3?1e-38*t():t()*4-1))}s.draw(2,0,3,f)}const u={version:4,bp:new Uint32Array(256),cp:new Uint32Array(256),xfMem:new Uint32Array(4096),xfRegs:new Uint32Array(96),texMem:new Uint8Array(0),frames:[]};return{stream:s.bytes(),mem:i,init:u}}function $i(e){const t=oi();let i;const n=new vi(()=>({mem:i.mem.data,tmem:i.regs.tmem,tluts:i.regs.tluts}));i=new ft(n,t.init,t.mem),n.drawing=!0,i.runFrame({fifo:t.stream,updates:[]});const s=n.efb,u=e.draws,r=e.drawing,f=e.fastForward;e.floatChecking_=!0;try{e.reset(),e.drawing=!0,e.fastForward=!1,new ft(e,t.init,t.mem.slice()).runFrame({fifo:t.stream,updates:[]}),e.flush();const a=e.readEfb();let c=0;for(let h=0;h<s.color.length;h++)(s.color[h]!==a.color[h]||s.depth[h]!==a.depth[h])&&c++;return c}finally{e.reset(),e.floatChecking_=!1,e.draws=u,e.drawing=r,e.fastForward=f}}const T=640,R=528,Zi=zi(),st=T/yt,mt=R/yt,Qi=4294967295,J=Ui/y;function ai(e){let t=0;const i=E(e[0],10,4)+1;if(E(e[0],16,3)>0)for(let n=0;n<i;n++)e[16+n]!==0&&(t|=1);return(E(e[241],21,3)>2||E(e[241],21,3)!==0&&E(e[232],10,1))&&(t|=2),t}function ne(e,t,i){let n=0;const s={drawing:!0,fastForward:!1,draw(r){n|=ai(r.bp)},copy(){return{toXfb:!1,rect:[],clear:!1}},copyClear(){}},u=new ft(s,e,t);for(const r of i)u.runFrame(r);return n}class se{constructor(t,i={},n={}){if(this.shaderOptions=i,this.gl=t,this.floatOps=!i.floatOps||i.floatOps==="auto"?"fast":i.floatOps,this.floatAuto=!i.floatOps||i.floatOps==="auto",(n.workers??0)>0&&It.available()&&(this.fp=new It(n.workers,n.factory??(()=>new Worker(new URL(""+new URL("frontworker.07b46661.js",import.meta.url).href,self.location),{type:"module"})))),t.getParameter(t.MAX_TEXTURE_SIZE)<J)throw new Error(`the GX renderer needs textures of ${J} texels (this GPU's limit is ${t.getParameter(t.MAX_TEXTURE_SIZE)})`);this.pending=new Map([["pixel",U(t,M,G(this.pixelOptions()))],["copy",U(t,M,Ii)],["clear",U(t,M,Di)],["picture",U(t,M,Ci)],["present",U(t,M,Pi)]]),this.mem=I(t,"R8UI",y,J),this.memFbo=this.fboOf(this.mem);for(let r=0;r<2;r++){const f=I(t,"RG32UI",T,R);this.efb.push({fbo:this.fboOf(f),tex:f})}const s=t.createTexture();t.bindTexture(t.TEXTURE_2D,s),t.texStorage2D(t.TEXTURE_2D,1,t.RGBA8,T,R);const u=t.createFramebuffer();t.bindFramebuffer(t.FRAMEBUFFER,u),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,s,0),this.pictureFbo={fbo:u,tex:s};for(let r=0;r<4;r++)this.sets.push({data:I(t,"RGBA32UI",1024,64),list:I(t,"R32UI",4096,64),tiles:I(t,"RG32UI",1024,Math.ceil(st*mt/1024))});this.statesTex=I(t,"R32UI",4096,64),this.timer={ext:t.getExtension("EXT_disjoint_timer_query_webgl2"),pending:[],ms:{}}}drawing=!0;fastForward=!1;onPicture=null;pictures=0;tris=0;draws=0;onFlush=null;passes=0;lastFeatures=0;stats={cpuMs:0,flushMs:0,uploadMs:0,memBytes:0,listEntries:0,tilesTouched:0};gl;pending;progs=new Map;get progPixel(){return this.program("pixel")}get progCopy(){return this.program("copy")}get progClear(){return this.program("clear")}get progPicture(){return this.program("picture")}needFeatures=0;stateFeatures=[];mem;memFbo;efb=[];cur=0;pictureFbo;havePicture=!1;sets=[];setNo=0;statesTex;q=new pt;states=new pt;stateCount=0;stateUploaded=0;stateIndex=new Map;stateRanges=[];lastStateVersion=[-1,-1,-1];lastState=0;palMap=new Map;palNext=0;attached=null;dirty=[];used=new Map;tmemVersion=-1;ctx=new Ot;fp=null;ent=new jt;uniforms=new Map;measure=!1;timing=!1;timer=null;syncPixel=new Uint32Array(4);sync(){const t=this.gl;t.readPixels(0,0,1,1,t.RGBA_INTEGER,t.UNSIGNED_INT,this.syncPixel)}timeBegin(){if(!this.timing)return 0;const t=this.timer;if(t.ext){const i=this.gl.createQuery();return this.gl.beginQuery(t.ext.TIME_ELAPSED_EXT,i),t.cur=i,0}return this.sync(),performance.now()}timeEnd(t,i){if(!this.timing||!this.timer)return;const n=this.timer;if(n.ext){this.gl.endQuery(n.ext.TIME_ELAPSED_EXT),n.pending.push({q:n.cur,what:i});return}this.sync(),n.ms[i]=(n.ms[i]??0)+performance.now()-t}collectTimes(){const t=this.timer;if(!t)return{};const i=this.gl;if(t.ext){i.finish();for(const s of t.pending){for(let u=0;u<1e3&&!i.getQueryParameter(s.q,i.QUERY_RESULT_AVAILABLE);u++);t.ms[s.what]=(t.ms[s.what]??0)+Number(i.getQueryParameter(s.q,i.QUERY_RESULT))/1e6,i.deleteQuery(s.q)}t.pending=[]}const n=t.ms;return t.ms={},n}reset(){const t=this.gl;this.attached&&(this.attached.onTouch=null,this.attached.beforeWrite=null),this.attached=null,this.drainFront(),this.q.n=0,this.ent.n=0,this.used.clear(),this.dirty=[],this.palMap.clear(),this.palNext=0,this.resetStates(),this.tmemVersion=-1;for(const i of this.efb)t.bindFramebuffer(t.FRAMEBUFFER,i.fbo),t.clearBufferuiv(t.COLOR,0,new Uint32Array([0,0,0,0]));this.cur=0,this.pictures=0,this.tris=0,this.passes=0}program(t){let i=this.progs.get(t);return i||(i=Wt(this.gl,this.pending.get(t)),this.progs.set(t,i),this.pending.delete(t)),i}floatOps;floatAuto;floatCheck=null;floatChecking=null;pixelOptions(){return{...this.shaderOptions,floatOps:this.floatOps}}checkFloat(){return!this.floatAuto||this.floatOps!=="fast"?Promise.resolve():(this.floatChecking||(this.floatChecking=(async()=>{let t=0,i=-1,n=-1,s;try{let u=performance.now();if(i=Yi(this.gl),t+=performance.now()-u,i===0){const r=this.pending.get("pixel");for(;r&&!ht(this.gl,r);)await new Promise(f=>setTimeout(f,10));u=performance.now(),n=$i(this),t+=performance.now()-u}}catch(u){s=String(u?.message??u)}this.floatCheck={ops:i,pixels:n,ms:t,error:s},(i!==0||n!==0)&&this.useSoftFloat()})()),this.floatChecking)}useSoftFloat(){this.floatOps="soft";for(const t of[...this.progs.keys()])t.startsWith("pixel")&&(this.gl.deleteProgram(this.progs.get(t)),this.progs.delete(t));for(const t of[...this.pending.keys()])t.startsWith("pixel")&&this.pending.delete(t);this.pending.set("pixel",U(this.gl,M,G(this.pixelOptions())))}warmUp(t=6){const i=oi(),n=this.draws,s=this.drawing,u=this.fastForward;try{this.reset(),this.drawing=!0,this.fastForward=!1;const r=new ft(this,i.init,i.mem);for(let f=0;f<t;f++)r.runFrame({fifo:i.stream,updates:[]}),this.flush();this.finish()}finally{this.reset(),this.draws=n,this.drawing=s,this.fastForward=u}}async prepare(t=0){await this.checkFloat();for(const i of[1,2,3])(t&i)===i&&!this.pending.has("pixel"+i)&&!this.progs.has("pixel"+i)&&this.pending.set("pixel"+i,U(this.gl,M,G({...this.pixelOptions(),indirect:(i&1)!==0,fogExtra:(i&2)!==0})));for(const[i,n]of[...this.pending]){for(;!ht(this.gl,n);)await new Promise(s=>setTimeout(s,10));this.program(i)}}async warm(t=3){await this.checkFloat();for(const i of[2,1,3]){if((t&i)!==i||this.progs.has("pixel"+i))continue;this.pending.has("pixel"+i)||this.pending.set("pixel"+i,U(this.gl,M,G({...this.pixelOptions(),indirect:(i&1)!==0,fogExtra:(i&2)!==0})));const n=this.pending.get("pixel"+i);for(;!ht(this.gl,n);)await new Promise(s=>setTimeout(s,50));this.pending.has("pixel"+i)&&this.program("pixel"+i)}}compileWaits={count:0,ms:0};pixelProgram(t){if(t===0)return this.progPixel;const i="pixel"+t;if(this.progs.has(i))return this.progs.get(i);this.pending.has(i)||this.pending.set(i,U(this.gl,M,G({...this.pixelOptions(),indirect:(t&1)!==0,fogExtra:(t&2)!==0})));const n=performance.now(),s=this.program(i);return this.compileWaits.count++,this.compileWaits.ms+=performance.now()-n,s}fboOf(t){const i=this.gl,n=i.createFramebuffer();return i.bindFramebuffer(i.FRAMEBUFFER,n),i.framebufferTexture2D(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,t.tex,0),n}u(t,i){let n=this.uniforms.get(t);return n||(n=new Map,this.uniforms.set(t,n)),n.has(i)||n.set(i,this.gl.getUniformLocation(t,i)),n.get(i)}bindTex(t,i,n,s){const u=this.gl;u.activeTexture(u.TEXTURE0+i),u.bindTexture(u.TEXTURE_2D,s.tex),u.uniform1i(this.u(t,n),i)}prepareMemory(t,i){this.attach(t,i)}attach(t,i){if(this.attached===t)return;this.attached=t;const n=this.gl;B(n,this.mem,0,0,y,t.data.length/y,t.data),this.uploadTmem(i),this.dirty=[],t.onTouch=(s,u)=>{this.dirty.push(s,u)},t.beforeWrite=(s,u)=>{for(const[r,f]of this.used)if(s<r+f&&r<s+u){this.flush();break}}}uploadTmem(t){B(this.gl,this.mem,0,Gt/y,y,t.tmem.length/y,t.tmem),this.tmemVersion=t.tmemVersion}uploadBytes(t,i,n,s){const u=this.gl,r=t+i;let f=t;if(f&y-1){const a=Math.min(r,(f|y-1)+1)-f;B(u,this.mem,f&y-1,f/y|0,a,1,n.subarray(f-s,f-s+a)),f+=a}const o=(r-f)/y|0;o>0&&(B(u,this.mem,0,f/y,y,o,n.subarray(f-s,f-s+o*y)),f+=o*y),f<r&&B(u,this.mem,0,f/y,r-f,1,n.subarray(f-s,r-s))}uploadDirty(t){if(!this.dirty.length)return;const i=this.dirty;this.dirty=[];for(let r=1;r<i.length;r+=2)this.stats.memBytes+=i[r];const n=[];for(let r=0;r<i.length;r+=2)n.push(r);n.sort((r,f)=>i[r]-i[f]);let s=-1,u=-1;for(const r of n){const f=i[r],o=i[r]+i[r+1];s<0?(s=f,u=o):f<=u?u=Math.max(u,o):(this.uploadBytes(s,u-s,t.data,0),s=f,u=o)}s>=0&&this.uploadBytes(s,u-s,t.data,0)}paletteAddress(t){let i=this.palMap.get(t);return i!==void 0||(this.palNext+t.length>qt&&(this.flush(),this.palMap.clear(),this.palNext=0,this.resetStates()),i=Xt+this.palNext,this.uploadBytes(i,t.length,t,i),this.palMap.set(t,i),this.palNext=this.palNext+t.length+31&-32),i}resetStates(){this.stateCount=0,this.stateUploaded=0,this.states.n=0,this.stateIndex.clear(),this.stateRanges=[],this.stateFeatures=[],this.lastStateVersion=[-1,-1,-1]}wScratch=new Uint32Array(z);textureRanges(t,i){const n=[],s=new Set;for(let u=0;u<i;u++){const r=t.bp[40+(u>>1)]>>>(u&1?12:0);if(!E(r,6,1))continue;const f=E(r,0,3);if(s.has(f))continue;s.add(f);const o=f>=4,a=f&3,c=t.bp[(o?168:136)+a],h=t.bp[(o?172:140)+a],p=t.bp[(o?180:148)+a],x=E(c,20,4),g=E(c,0,10),d=E(c,10,10);if(E(h,21,1))continue;let w=1,l=g+1,b=d+1;for(;l>1||b>1;)l>>=1,b>>=1,w++;const m=ki(x),A=yi(x,g,d,w-1)+Math.max(l,m.bw)*Math.max(b,m.bh)*m.nibbles/2+64;n.push((p&16777215)*32,A)}return n}stateFor(t){this.stateCount>2e4&&(this.flush(),this.palMap.clear(),this.palNext=0,this.resetStates());const i=t.tmemVersion;if(this.lastStateVersion[0]===t.bpVersion&&this.lastStateVersion[1]===t.xfVersion&&this.lastStateVersion[2]===i)return this.lastState;const n=this.wScratch;n.fill(0),n.set(t.bp);for(let c=0;c<8;c++)n[224+c]=t.tevBank[0][c];for(let c=0;c<8;c++)n[256+c]=t.tevBank[1][c];let s=0;for(let c=0;c<8;c++)t.xf[4160+c]>>1&1&&(s|=1<<c);n[272]=s,n[273]=t.xf[4122];const u=E(t.bp[0],10,4)+1;let r=0;for(let c=0;c<8;c++)n[264+c]=Qi;for(let c=0;c<u;c++){const h=t.bp[40+(c>>1)]>>>(c&1?12:0);if(!(h>>6&1))continue;const p=h&7;if(r>>p&1)continue;r|=1<<p;const x=p>=4,g=p&3,d=t.bp[(x?168:136)+g]>>20&15;if(d===8||d===9||d===10){const w=t.tluts.get(t.bp[(x?184:152)+g]&1023);w&&(n[264+p]=this.paletteAddress(w),n[274+p]=w.length)}}let f=2166136261;for(let c=0;c<z;c++)f^=n[c],f=Math.imul(f,16777619);f>>>=0;let o=-1;const a=this.stateIndex.get(f);if(a)for(const c of a){const h=c*z;let p=!0;for(let x=0;x<z;x++)if(this.states.words[h+x]!==n[x]){p=!1;break}if(p){o=c;break}}return o<0&&(o=this.stateCount++,this.states.ensure(z),this.states.words.set(n,o*z),this.states.n=this.stateCount*z,a?a.push(o):this.stateIndex.set(f,[o]),this.stateRanges[o]=this.textureRanges(t,u),this.stateFeatures[o]=ai(t.bp)),this.lastStateVersion=[t.bpVersion,t.xfVersion,i],this.lastState=o,o}draw(t,i,n,s,u,r,f){if(!this.drawing||this.fastForward)return;this.draws++;const o=this.measure?performance.now():0;this.attach(i,t),t.tmemVersion!==this.tmemVersion&&(this.flush(),this.uploadTmem(t));const a=this.stateFor(t),c=this.stateRanges[a];this.needFeatures|=this.stateFeatures[a];for(let h=0;h<c.length;h+=2)(this.used.get(c[h])??0)<c[h+1]&&this.used.set(c[h],c[h+1]);if(this.fp&&!Si(t.bp,f))this.fp.submit(t,n,f,a,()=>this.drainFront());else{this.drainFront();const h=this.q,p=h.n;this.ctx.run(t,n,f,a,h),this.tris+=dt(h.ints.subarray(p),h.words.subarray(p),h.n-p,p>>2,this.ent)}this.measure&&(this.stats.cpuMs+=performance.now()-o)}drainFront(){const t=this.fp;if(!t||!t.busy)return;const i=this.q,n=this.ent;t.drain((s,u,r)=>{i.ensure(s.length);const f=i.n;if(i.words.set(s,f),i.n+=s.length,!u){this.tris+=dt(i.ints.subarray(f),i.words.subarray(f),s.length,f>>2,n);return}this.tris+=r,n.ensure(u.length);const o=n.a,a=f>>2;let c=n.n;for(let h=0;h<u.length;h+=2)o[c++]=u[h],o[c++]=u[h+1]+a;n.n=c}),t.zFresh&&(t.zFresh=!1,_i(t.zPlane))}get poolStats(){return this.fp?{...this.fp.stats,workerJobs:this.fp.workerJobs,workersStarted:this.fp.workersStarted}:null}dispose(){this.drainFront(),this.fp?.dispose(),this.fp=null}grow(t,i,n){const s=Math.ceil(i/n);if(s<=t.h)return t;let u=t.h;for(;u<s;)u*=2;return this.gl.deleteTexture(t.tex),I(this.gl,t.fmt,t.w,u)}floatChecking_=!1;flush(){this.drainFront();const t=this.q.n,i=this.measure?performance.now():0;if(this.attached&&this.uploadDirty(this.attached),this.measure&&(this.stats.uploadMs+=performance.now()-i),t===0){this.used.clear();return}this.floatAuto&&this.floatOps==="fast"&&!this.floatCheck&&!this.floatChecking_&&this.useSoftFloat();const n=this.measure?performance.now():0,s=this.gl,u=st*mt,r=new Uint32Array(u+1),f=this.ent.a,o=this.ent.n;for(let v=0;v<o;v+=2)r[f[v]]++;const a=new Uint32Array(u+1);let c=0;for(let v=0;v<u;v++)a[v]=c,c+=r[v];this.stats.listEntries+=c;const h=new Uint32Array(Math.max(1,Math.ceil(c/4096))*4096),p=new Uint32Array(u);for(let v=0;v<o;v+=2){const C=f[v];h[a[C]+p[C]++]=f[v+1]}const x=Math.ceil(u/1024),g=new Uint32Array(1024*x*2);for(let v=0;v<st*mt;v++)g[v*2]=a[v],g[v*2+1]=r[v];const d=this.sets[this.setNo++%this.sets.length];d.data=this.grow(d.data,this.q.n,4096);const w=Math.ceil(this.q.n/4096);B(s,d.data,0,0,1024,w,this.q.words.subarray(0,w*4096)),d.list=this.grow(d.list,h.length,4096),B(s,d.list,0,0,4096,h.length/4096,h),B(s,d.tiles,0,0,1024,x,g);const l=this.statesTex;this.statesTex=this.grow(this.statesTex,this.states.n,4096),this.statesTex!==l&&(this.stateUploaded=0);const b=Math.ceil(this.states.n/4096),m=Math.floor(this.stateUploaded*z/4096);this.states.ensure(b*4096-this.states.n),B(s,this.statesTex,0,m,4096,b-m,this.states.words.subarray(m*4096,b*4096)),this.stateUploaded=this.stateCount;const A=this.efb[this.cur],_=this.efb[1-this.cur];s.bindFramebuffer(s.FRAMEBUFFER,_.fbo),s.viewport(0,0,T,R),s.disable(s.SCISSOR_TEST);const k=this.pixelProgram(this.needFeatures);this.lastFeatures=this.needFeatures,s.useProgram(k),this.bindTex(k,0,"uMem",this.mem),this.bindTex(k,1,"uData",d.data),this.bindTex(k,2,"uTiles",d.tiles),this.bindTex(k,3,"uList",d.list),this.bindTex(k,4,"uStates",this.statesTex),this.bindTex(k,5,"uEfb",A.tex),s.uniform1i(this.u(k,"uTilesX"),st),s.uniform4i(this.u(k,"uK"),4,25,3,2),this.needFeatures&2&&s.uniform4uiv(this.u(k,"uExpC"),Zi);const S=this.timeBegin();s.drawArrays(s.TRIANGLES,0,3),this.timeEnd(S,"pixel"),this.cur=1-this.cur,this.passes++,this.q.n=0,this.ent.n=0,this.used.clear(),this.needFeatures=0,this.measure&&(this.stats.flushMs+=performance.now()-n),this.onFlush?.()}finish(){this.gl.finish()}copy(t,i,n){this.attach(i,t),this.flush();const s=this.gl,u=kt(t.bp),r=this.efb[this.cur];if(u.toScreen)this.drawPicture(u.pixFmt),this.havePicture=!0,this.onPicture&&this.onPicture(this.readKept(),this.pictures),this.pictures++;else{const f=Ai(u),o={I4:0,I8:1,IA4:2,IA8:3,RGB565:4,RGB5A3:5,RGBA8:6,R4:7,Z16:16,Z16L:17,RA4:8,RA8:9,A8:10,R8:11,G8:12,B8:13,RG8:14,GB8:15};if(f!=="none"){const a={I4:[3,3,32],I8:[3,2,32],IA4:[3,2,32],IA8:[2,2,32],RGB565:[2,2,32],RGB5A3:[2,2,32],RGBA8:[2,2,64],R4:[3,3,32],Z16:[2,2,32],Z16L:[2,2,32],RA4:[3,2,32],RA8:[2,2,32],A8:[3,2,32],R8:[3,2,32],G8:[3,2,32],B8:[3,2,32],RG8:[2,2,32],GB8:[2,2,32]},[c,h,p]=a[f],x=u.half?1:0,g=(u.wm>>x>>c)+1,d=(u.hm>>x>>h)+1,w=u.dest&33554431,l=(d-1)*u.stride+g*p;s.bindFramebuffer(s.FRAMEBUFFER,this.memFbo),s.viewport(0,0,y,J),s.enable(s.SCISSOR_TEST);const b=w/y|0,m=Math.min(J-1,(w+l)/y|0);s.scissor(0,b,y,m-b+1);const A=this.progCopy;s.useProgram(A),this.bindTex(A,0,"uEfb",r.tex);const _=(S,v)=>s.uniform1i(this.u(A,S),v);_("uDest",w),_("uStride",u.stride),_("uFmt",o[f]),_("uBlocksX",g),_("uBlocksY",d),_("uKw",c),_("uKh",h),_("uBytes",p),_("uHalf",x),_("uX0",u.x0),_("uY0",u.y0),_("uPixFmt",u.pixFmt);const k=this.timeBegin();s.drawArrays(s.TRIANGLES,0,3),this.timeEnd(k,"copy"),s.disable(s.SCISSOR_TEST)}}return{toXfb:u.toScreen,rect:[u.x0,u.y0,u.wm+1,u.hm+1],clear:u.clear}}copyClear(t){this.flush();const i=this.gl,n=kt(t.bp),s=t.bp,u=s[65],r=s[79]&255,f=s[79]>>8&255,o=s[80]&255,a=s[80]>>8&255,c=this.efb[this.cur],h=this.efb[1-this.cur];i.bindFramebuffer(i.FRAMEBUFFER,h.fbo),i.viewport(0,0,T,R),i.disable(i.SCISSOR_TEST);const p=this.progClear;i.useProgram(p),this.bindTex(p,0,"uEfb",c.tex),i.uniform4i(this.u(p,"uRect"),n.x0,n.y0,n.x0+n.wm,n.y0+n.hm);const x=n.pixFmt===1?(o>>2<<6|a>>2<<12|r>>2<<18)>>>0:(o|a<<8|r<<16)>>>0;i.uniform1ui(this.u(p,"uColor"),x),i.uniform1ui(this.u(p,"uAlpha6"),f>>2),i.uniform1ui(this.u(p,"uDepth"),s[81]&16777215),i.uniform1i(this.u(p,"uColorUpd"),u>>3&1),i.uniform1i(this.u(p,"uAlphaUpd"),u>>4&1),i.uniform1i(this.u(p,"uZUpd"),E(s[64],4,1)),i.uniform1i(this.u(p,"uPixFmt"),n.pixFmt),i.drawArrays(i.TRIANGLES,0,3),this.cur=1-this.cur}writeEfb(t,i){const n=new Uint32Array(T*R*2);for(let s=0;s<t.length;s++)n[s*2]=t[s],n[s*2+1]=i[s];B(this.gl,this.efb[this.cur].tex,0,0,T,R,n)}readMem(t,i){const n=this.gl;n.bindFramebuffer(n.FRAMEBUFFER,this.memFbo);const s=t/y|0,u=(t+i-1)/y|0,r=new Uint32Array(y*(u-s+1)*4);n.readPixels(0,s,y,u-s+1,n.RGBA_INTEGER,n.UNSIGNED_INT,r);const f=new Uint8Array(i);for(let o=0;o<i;o++)f[o]=r[(t-s*y+o)*4];return f}readEfb(){const t=this.gl;t.bindFramebuffer(t.FRAMEBUFFER,this.efb[this.cur].fbo);const i=new Uint32Array(T*R*4);t.readPixels(0,0,T,R,t.RGBA_INTEGER,t.UNSIGNED_INT,i);const n=new Uint32Array(T*R),s=new Uint32Array(T*R);for(let u=0;u<n.length;u++)n[u]=i[u*4],s[u]=i[u*4+1];return{color:n,depth:s}}present(t,i,n=0){if(t!==void 0&&typeof t!="number"&&t!==this.gl.canvas){const f="putImageData"in t?t:t.getContext("2d"),o=this.readPicture(n);f?.putImageData(new ImageData(new Uint8ClampedArray(o.buffer,o.byteOffset,o.length),640,480),0,0);return}const s=typeof t=="number"?t:this.gl.drawingBufferWidth;i=typeof t=="number"?i:this.gl.drawingBufferHeight,this.flush();const u=this.gl;if(this.havePicture){u.bindFramebuffer(u.READ_FRAMEBUFFER,this.pictureFbo.fbo),u.bindFramebuffer(u.DRAW_FRAMEBUFFER,null),u.disable(u.SCISSOR_TEST),u.blitFramebuffer(0,0,T,480,0,i,s,0,u.COLOR_BUFFER_BIT,u.NEAREST),u.bindFramebuffer(u.FRAMEBUFFER,null);return}u.bindFramebuffer(u.FRAMEBUFFER,null),u.viewport(0,0,s,i),u.disable(u.SCISSOR_TEST);const r=this.program("present");u.useProgram(r),this.bindTex(r,0,"uEfb",this.efb[this.cur].tex),u.uniform1i(this.u(r,"uPixFmt"),n),u.uniform2f(this.u(r,"uSize"),s,i),u.drawArrays(u.TRIANGLES,0,3)}readPicture(t){return this.drawPicture(t),this.readKept()}picture(){return this.havePicture?this.readKept():null}readKept(){const t=this.gl;t.bindFramebuffer(t.FRAMEBUFFER,this.pictureFbo.fbo);const i=new Uint8Array(T*480*4);return t.readPixels(0,0,T,480,t.RGBA,t.UNSIGNED_BYTE,i),i}drawPicture(t){const i=this.gl;i.bindFramebuffer(i.FRAMEBUFFER,this.pictureFbo.fbo),i.viewport(0,0,T,R),i.disable(i.SCISSOR_TEST);const n=this.progPicture;i.useProgram(n),this.bindTex(n,0,"uEfb",this.efb[this.cur].tex),i.uniform1i(this.u(n,"uPixFmt"),t),i.drawArrays(i.TRIANGLES,0,3)}}export{se as GxGl,ai as drawFeatures,ne as scanFeatures};
