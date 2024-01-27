/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:54:50 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Js_toe_roll_joint_right_src.h"

#ifdef _MSC_VER
  #define INLINE __forceinline /* use __forceinline (VC++ specific) */
#else
  #define INLINE inline        /* use standard inline */
#endif

/**
 * Copied from Wolfram Mathematica C Definitions file mdefs.hpp
 * Changed marcos to inline functions (Eric Cousineau)
 */
INLINE double Power(double x, double y) { return pow(x, y); }
INLINE double Sqrt(double x) { return sqrt(x); }

INLINE double Abs(double x) { return fabs(x); }

INLINE double Exp(double x) { return exp(x); }
INLINE double Log(double x) { return log(x); }

INLINE double Sin(double x) { return sin(x); }
INLINE double Cos(double x) { return cos(x); }
INLINE double Tan(double x) { return tan(x); }

INLINE double Csc(double x) { return 1.0/sin(x); }
INLINE double Sec(double x) { return 1.0/cos(x); }

INLINE double ArcSin(double x) { return asin(x); }
INLINE double ArcCos(double x) { return acos(x); }
//INLINE double ArcTan(double x) { return atan(x); }

/* update ArcTan function to use atan2 instead. */
INLINE double ArcTan(double x, double y) { return atan2(y,x); }

INLINE double Sinh(double x) { return sinh(x); }
INLINE double Cosh(double x) { return cosh(x); }
INLINE double Tanh(double x) { return tanh(x); }

#define E 2.71828182845904523536029
#define Pi 3.14159265358979323846264
#define Degree 0.01745329251994329576924

/*
 * Sub functions
 */
static void output1(double *p_output1,const double *var1)
{
  double t48;
  double t68;
  double t86;
  double t96;
  double t184;
  double t188;
  double t186;
  double t189;
  double t195;
  double t313;
  double t298;
  double t322;
  double t330;
  double t335;
  double t216;
  double t218;
  double t219;
  double t400;
  double t401;
  double t405;
  double t407;
  double t408;
  double t410;
  double t412;
  double t413;
  double t417;
  double t403;
  double t418;
  double t420;
  double t428;
  double t429;
  double t437;
  double t441;
  double t459;
  double t460;
  double t461;
  double t488;
  double t491;
  double t494;
  double t377;
  double t380;
  double t399;
  double t37;
  double t308;
  double t337;
  double t343;
  double t478;
  double t481;
  double t482;
  double t538;
  double t540;
  double t542;
  double t550;
  double t551;
  double t552;
  double t357;
  double t358;
  double t372;
  double t584;
  double t586;
  double t590;
  double t593;
  double t608;
  double t616;
  double t629;
  double t717;
  double t719;
  double t738;
  double t696;
  double t848;
  double t872;
  double t881;
  double t887;
  double t786;
  double t794;
  double t705;
  double t711;
  double t750;
  double t766;
  double t908;
  double t910;
  double t917;
  double t918;
  double t936;
  double t944;
  double t947;
  double t956;
  double t973;
  double t979;
  double t981;
  double t996;
  double t920;
  double t962;
  double t1002;
  double t1015;
  double t820;
  double t823;
  double t1054;
  double t1055;
  double t1057;
  double t1058;
  double t1024;
  double t1029;
  double t1032;
  double t1034;
  double t1089;
  double t1092;
  double t1093;
  double t1095;
  double t842;
  double t845;
  double t714;
  double t742;
  double t767;
  double t768;
  double t1037;
  double t1041;
  double t1042;
  double t1053;
  double t1172;
  double t1174;
  double t1177;
  double t1194;
  double t807;
  double t828;
  double t829;
  double t836;
  double t1098;
  double t1110;
  double t1113;
  double t1114;
  double t1212;
  double t1219;
  double t1221;
  double t1224;
  double t1293;
  double t1297;
  double t1299;
  double t1305;
  double t1313;
  double t1314;
  double t1319;
  double t1327;
  double t1387;
  double t847;
  double t877;
  double t891;
  double t902;
  double t1410;
  double t1414;
  double t1418;
  double t1432;
  double t1436;
  double t1440;
  double t1442;
  double t1269;
  double t1278;
  double t1280;
  double t1284;
  double t1473;
  double t1476;
  double t1395;
  double t1398;
  double t1389;
  double t1390;
  double t1147;
  double t1149;
  double t1150;
  double t1154;
  double t1511;
  double t1512;
  double t1519;
  double t1523;
  double t1531;
  double t1532;
  double t1534;
  double t1535;
  double t1542;
  double t1545;
  double t1549;
  double t1551;
  double t1463;
  double t1465;
  double t1525;
  double t1537;
  double t1554;
  double t1556;
  double t1558;
  double t1561;
  double t1562;
  double t1568;
  double t1422;
  double t1423;
  double t1575;
  double t1578;
  double t1581;
  double t1583;
  double t1614;
  double t1615;
  double t1617;
  double t1618;
  double t1493;
  double t1500;
  double t1505;
  double t1508;
  double t1437;
  double t1438;
  double t1443;
  double t1448;
  double t1593;
  double t1595;
  double t1608;
  double t1609;
  double t1666;
  double t1667;
  double t1668;
  double t1669;
  double t1682;
  double t1686;
  double t1687;
  double t1689;
  double t1455;
  double t1466;
  double t1486;
  double t1490;
  double t1749;
  double t1752;
  double t1758;
  double t1763;
  double t1771;
  double t1776;
  double t1777;
  double t1778;
  double t1392;
  double t1399;
  double t1424;
  double t1426;
  double t1842;
  double t1854;
  double t1855;
  double t1828;
  double t1915;
  double t1918;
  double t1924;
  double t1927;
  double t1726;
  double t1730;
  double t1734;
  double t1739;
  double t1885;
  double t1886;
  double t1833;
  double t1834;
  double t1638;
  double t1646;
  double t1649;
  double t1651;
  double t1860;
  double t1861;
  double t1936;
  double t1938;
  double t1943;
  double t1946;
  double t1950;
  double t1961;
  double t1968;
  double t1974;
  double t1981;
  double t1990;
  double t1995;
  double t2003;
  double t1948;
  double t1978;
  double t2008;
  double t2023;
  double t1889;
  double t1890;
  double t2051;
  double t2059;
  double t2060;
  double t2063;
  double t2024;
  double t2025;
  double t2026;
  double t2029;
  double t2073;
  double t2081;
  double t2082;
  double t2085;
  double t1903;
  double t1909;
  double t1835;
  double t1858;
  double t1864;
  double t1877;
  double t2040;
  double t2041;
  double t2045;
  double t2050;
  double t2151;
  double t2155;
  double t2157;
  double t2158;
  double t1887;
  double t1895;
  double t1897;
  double t1899;
  double t2097;
  double t2101;
  double t2110;
  double t2113;
  double t2162;
  double t2164;
  double t2175;
  double t2176;
  double t2230;
  double t2232;
  double t2233;
  double t2235;
  double t2247;
  double t2249;
  double t2251;
  double t2273;
  double t2319;
  double t1910;
  double t1923;
  double t1930;
  double t1931;
  double t2329;
  double t2331;
  double t2333;
  double t2365;
  double t2370;
  double t2382;
  double t2383;
  double t2222;
  double t2223;
  double t2224;
  double t2225;
  double t2424;
  double t2428;
  double t2324;
  double t2326;
  double t2320;
  double t2321;
  double t2135;
  double t2136;
  double t2138;
  double t2140;
  double t2451;
  double t2461;
  double t2463;
  double t2464;
  double t2476;
  double t2481;
  double t2486;
  double t2487;
  double t2490;
  double t2492;
  double t2496;
  double t2497;
  double t2419;
  double t2420;
  double t2475;
  double t2488;
  double t2498;
  double t2500;
  double t2503;
  double t2505;
  double t2507;
  double t2512;
  double t2334;
  double t2336;
  double t2516;
  double t2518;
  double t2522;
  double t2526;
  double t2555;
  double t2556;
  double t2558;
  double t2560;
  double t2440;
  double t2448;
  double t2449;
  double t2450;
  double t2371;
  double t2376;
  double t2387;
  double t2391;
  double t2546;
  double t2547;
  double t2548;
  double t2549;
  double t2589;
  double t2591;
  double t2593;
  double t2595;
  double t2604;
  double t2607;
  double t2610;
  double t2613;
  double t2399;
  double t2422;
  double t2432;
  double t2433;
  double t2665;
  double t2667;
  double t2668;
  double t2672;
  double t2679;
  double t2683;
  double t2685;
  double t2686;
  double t2322;
  double t2328;
  double t2338;
  double t2344;
  double t2795;
  double t2799;
  double t2801;
  double t2776;
  double t2858;
  double t2859;
  double t2854;
  double t2856;
  double t2653;
  double t2654;
  double t2656;
  double t2659;
  double t2823;
  double t2827;
  double t2780;
  double t2783;
  double t2568;
  double t2573;
  double t2574;
  double t2575;
  double t2807;
  double t2812;
  double t2868;
  double t2877;
  double t2880;
  double t2881;
  double t2885;
  double t2887;
  double t2888;
  double t2890;
  double t2894;
  double t2896;
  double t2897;
  double t2900;
  double t2882;
  double t2891;
  double t2907;
  double t2911;
  double t2829;
  double t2837;
  double t2949;
  double t2950;
  double t2955;
  double t2958;
  double t2850;
  double t2851;
  double t2915;
  double t2917;
  double t2920;
  double t2927;
  double t2976;
  double t2980;
  double t2981;
  double t2982;
  double t3044;
  double t3047;
  double t3048;
  double t3050;
  double t2788;
  double t2803;
  double t2818;
  double t2820;
  double t2942;
  double t2945;
  double t2946;
  double t2947;
  double t3067;
  double t3068;
  double t3072;
  double t3078;
  double t2828;
  double t2838;
  double t2839;
  double t2841;
  double t2988;
  double t2991;
  double t2994;
  double t3006;
  double t3082;
  double t3098;
  double t3100;
  double t3104;
  double t3027;
  double t3033;
  double t3038;
  double t3039;
  double t2852;
  double t2857;
  double t2860;
  double t2863;
  double t3176;
  double t3177;
  double t3181;
  double t3184;
  double t3187;
  double t3188;
  double t3194;
  double t3197;
  t48 = Cos(var1[3]);
  t68 = Sin(var1[3]);
  t86 = Cos(var1[4]);
  t96 = Sin(var1[4]);
  t184 = Cos(var1[5]);
  t188 = Sin(var1[5]);
  t186 = t48*t184*t96;
  t189 = t68*t188;
  t195 = t186 + t189;
  t313 = Cos(var1[17]);
  t298 = Sin(var1[17]);
  t322 = -1.*t184*t68;
  t330 = t48*t96*t188;
  t335 = t322 + t330;
  t216 = t184*t68*t96;
  t218 = -1.*t48*t188;
  t219 = t216 + t218;
  t400 = -0.091*t298;
  t401 = 0. + t400;
  t405 = -1.*t313;
  t407 = 1. + t405;
  t408 = -0.091*t407;
  t410 = 0. + t408;
  t412 = t48*t184;
  t413 = t68*t96*t188;
  t417 = t412 + t413;
  t403 = t401*t219;
  t418 = t410*t417;
  t420 = 0. + var1[1] + t403 + t418;
  t428 = -1.*var1[2];
  t429 = -1.*t86*t184*t401;
  t437 = -1.*t410*t86*t188;
  t441 = 0. + t428 + t429 + t437;
  t459 = -1.*t298*t219;
  t460 = t313*t417;
  t461 = t459 + t460;
  t488 = t313*t219;
  t491 = t298*t417;
  t494 = t488 + t491;
  t377 = -1.*t86*t184*t298;
  t380 = t313*t86*t188;
  t399 = t377 + t380;
  t37 = -1.*var1[0];
  t308 = -1.*t298*t195;
  t337 = t313*t335;
  t343 = t308 + t337;
  t478 = t313*t86*t184;
  t481 = t86*t298*t188;
  t482 = t478 + t481;
  t538 = -1.*t401*t195;
  t540 = -1.*t410*t335;
  t542 = 0. + t37 + t538 + t540;
  t550 = t86*t184*t401;
  t551 = t410*t86*t188;
  t552 = 0. + var1[2] + t550 + t551;
  t357 = t313*t195;
  t358 = t298*t335;
  t372 = t357 + t358;
  t584 = -1.*var1[1];
  t586 = -1.*t401*t219;
  t590 = -1.*t410*t417;
  t593 = 0. + t584 + t586 + t590;
  t608 = t401*t195;
  t616 = t410*t335;
  t629 = 0. + var1[0] + t608 + t616;
  t717 = Cos(var1[18]);
  t719 = -1.*t717;
  t738 = 1. + t719;
  t696 = Sin(var1[18]);
  t848 = -0.930418*t696;
  t872 = 0. + t848;
  t881 = 0.366501*t696;
  t887 = 0. + t881;
  t786 = 0.930418*t696;
  t794 = 0. + t786;
  t705 = -0.366501*t696;
  t711 = 0. + t705;
  t750 = -0.134322983001*t738;
  t766 = 1. + t750;
  t908 = -0.04500040093286238*t738;
  t910 = -0.07877663122399998*t872;
  t917 = 0.031030906668*t887;
  t918 = 0. + t908 + t910 + t917;
  t936 = 1.296332362046933e-7*var1[18];
  t944 = -0.07877668146182712*t738;
  t947 = -0.045000372235*t794;
  t956 = t936 + t944 + t947;
  t973 = 3.2909349868922137e-7*var1[18];
  t979 = 0.03103092645718495*t738;
  t981 = -0.045000372235*t711;
  t996 = t973 + t979 + t981;
  t920 = t918*t96;
  t962 = -1.*t956*t399;
  t1002 = -1.*t996*t482;
  t1015 = 0. + t428 + t429 + t920 + t437 + t962 + t1002;
  t820 = -0.8656776547239999*t738;
  t823 = 1. + t820;
  t1054 = t86*t918*t68;
  t1055 = t956*t461;
  t1057 = t996*t494;
  t1058 = 0. + var1[1] + t1054 + t403 + t418 + t1055 + t1057;
  t1024 = t86*t711*t68;
  t1029 = 0.340999127418*t738*t461;
  t1032 = t766*t494;
  t1034 = t1024 + t1029 + t1032;
  t1089 = t86*t794*t68;
  t1092 = t823*t461;
  t1093 = 0.340999127418*t738*t494;
  t1095 = t1089 + t1092 + t1093;
  t842 = -1.000000637725*t738;
  t845 = 1. + t842;
  t714 = t48*t86*t711;
  t742 = 0.340999127418*t738*t343;
  t767 = t766*t372;
  t768 = t714 + t742 + t767;
  t1037 = -1.*t711*t96;
  t1041 = 0.340999127418*t738*t399;
  t1042 = t766*t482;
  t1053 = t1037 + t1041 + t1042;
  t1172 = -1.*t918*t96;
  t1174 = t956*t399;
  t1177 = t996*t482;
  t1194 = 0. + var1[2] + t550 + t1172 + t551 + t1174 + t1177;
  t807 = t48*t86*t794;
  t828 = t823*t343;
  t829 = 0.340999127418*t738*t372;
  t836 = t807 + t828 + t829;
  t1098 = -1.*t794*t96;
  t1110 = t823*t399;
  t1113 = 0.340999127418*t738*t482;
  t1114 = t1098 + t1110 + t1113;
  t1212 = -1.*t48*t86*t918;
  t1219 = -1.*t956*t343;
  t1221 = -1.*t996*t372;
  t1224 = 0. + t37 + t1212 + t538 + t540 + t1219 + t1221;
  t1293 = t48*t86*t918;
  t1297 = t956*t343;
  t1299 = t996*t372;
  t1305 = 0. + var1[0] + t1293 + t608 + t616 + t1297 + t1299;
  t1313 = -1.*t86*t918*t68;
  t1314 = -1.*t956*t461;
  t1319 = -1.*t996*t494;
  t1327 = 0. + t584 + t1313 + t586 + t590 + t1314 + t1319;
  t1387 = Sin(var1[19]);
  t847 = t845*t48*t86;
  t877 = t872*t343;
  t891 = t887*t372;
  t902 = t847 + t877 + t891;
  t1410 = Cos(var1[19]);
  t1414 = -1.*t1410;
  t1418 = 1. + t1414;
  t1432 = -0.8656776547239999*t1418;
  t1436 = 1. + t1432;
  t1440 = -0.930418*t1387;
  t1442 = 0. + t1440;
  t1269 = -1.*t845*t96;
  t1278 = t872*t399;
  t1280 = t887*t482;
  t1284 = t1269 + t1278 + t1280;
  t1473 = -0.366501*t1387;
  t1476 = 0. + t1473;
  t1395 = 0.366501*t1387;
  t1398 = 0. + t1395;
  t1389 = 0.930418*t1387;
  t1390 = 0. + t1389;
  t1147 = t845*t86*t68;
  t1149 = t872*t461;
  t1150 = t887*t494;
  t1154 = t1147 + t1149 + t1150;
  t1511 = -1.296332362046933e-7*var1[19];
  t1512 = -0.14128592423750855*t1418;
  t1519 = -0.045000372235*t1442;
  t1523 = t1511 + t1512 + t1519;
  t1531 = 3.2909349868922137e-7*var1[19];
  t1532 = -0.055653945343889656*t1418;
  t1534 = -0.045000372235*t1476;
  t1535 = t1531 + t1532 + t1534;
  t1542 = -0.04500040093286238*t1418;
  t1545 = -0.055653909852*t1398;
  t1549 = -0.141285834136*t1390;
  t1551 = 0. + t1542 + t1545 + t1549;
  t1463 = -0.134322983001*t1418;
  t1465 = 1. + t1463;
  t1525 = t1523*t1034;
  t1537 = t1535*t1095;
  t1554 = t1551*t1154;
  t1556 = 0. + var1[1] + t1054 + t403 + t418 + t1055 + t1057 + t1525 + t1537 + t1554;
  t1558 = -1.*t1523*t1053;
  t1561 = -1.*t1535*t1114;
  t1562 = -1.*t1551*t1284;
  t1568 = 0. + t428 + t429 + t920 + t437 + t962 + t1002 + t1558 + t1561 + t1562;
  t1422 = -1.000000637725*t1418;
  t1423 = 1. + t1422;
  t1575 = t1436*t1034;
  t1578 = -0.340999127418*t1418*t1095;
  t1581 = t1442*t1154;
  t1583 = t1575 + t1578 + t1581;
  t1614 = -0.340999127418*t1418*t1034;
  t1615 = t1465*t1095;
  t1617 = t1476*t1154;
  t1618 = t1614 + t1615 + t1617;
  t1493 = t1436*t1053;
  t1500 = -0.340999127418*t1418*t1114;
  t1505 = t1442*t1284;
  t1508 = t1493 + t1500 + t1505;
  t1437 = t1436*t768;
  t1438 = -0.340999127418*t1418*t836;
  t1443 = t1442*t902;
  t1448 = t1437 + t1438 + t1443;
  t1593 = -0.340999127418*t1418*t1053;
  t1595 = t1465*t1114;
  t1608 = t1476*t1284;
  t1609 = t1593 + t1595 + t1608;
  t1666 = -1.*t1523*t768;
  t1667 = -1.*t1535*t836;
  t1668 = -1.*t1551*t902;
  t1669 = 0. + t37 + t1212 + t538 + t540 + t1219 + t1221 + t1666 + t1667 + t1668;
  t1682 = t1523*t1053;
  t1686 = t1535*t1114;
  t1687 = t1551*t1284;
  t1689 = 0. + var1[2] + t550 + t1172 + t551 + t1174 + t1177 + t1682 + t1686 + t1687;
  t1455 = -0.340999127418*t1418*t768;
  t1466 = t1465*t836;
  t1486 = t1476*t902;
  t1490 = t1455 + t1466 + t1486;
  t1749 = -1.*t1523*t1034;
  t1752 = -1.*t1535*t1095;
  t1758 = -1.*t1551*t1154;
  t1763 = 0. + t584 + t1313 + t586 + t590 + t1314 + t1319 + t1749 + t1752 + t1758;
  t1771 = t1523*t768;
  t1776 = t1535*t836;
  t1777 = t1551*t902;
  t1778 = 0. + var1[0] + t1293 + t608 + t616 + t1297 + t1299 + t1771 + t1776 + t1777;
  t1392 = t1390*t768;
  t1399 = t1398*t836;
  t1424 = t1423*t902;
  t1426 = t1392 + t1399 + t1424;
  t1842 = Cos(var1[20]);
  t1854 = -1.*t1842;
  t1855 = 1. + t1854;
  t1828 = Sin(var1[20]);
  t1915 = -0.930418*t1828;
  t1918 = 0. + t1915;
  t1924 = -0.366501*t1828;
  t1927 = 0. + t1924;
  t1726 = t1390*t1053;
  t1730 = t1398*t1114;
  t1734 = t1423*t1284;
  t1739 = t1726 + t1730 + t1734;
  t1885 = 0.930418*t1828;
  t1886 = 0. + t1885;
  t1833 = 0.366501*t1828;
  t1834 = 0. + t1833;
  t1638 = t1390*t1034;
  t1646 = t1398*t1095;
  t1649 = t1423*t1154;
  t1651 = t1638 + t1646 + t1649;
  t1860 = -0.134322983001*t1855;
  t1861 = 1. + t1860;
  t1936 = 0.039853038461262744*t1855;
  t1938 = -0.22023459268999998*t1918;
  t1943 = -0.086752619205*t1927;
  t1946 = 0. + t1936 + t1938 + t1943;
  t1950 = 6.295460977284962e-8*var1[20];
  t1961 = -0.22023473313910558*t1855;
  t1968 = 0.039853013046*t1886;
  t1974 = t1950 + t1961 + t1968;
  t1981 = -1.5981976069815686e-7*var1[20];
  t1990 = -0.08675267452931407*t1855;
  t1995 = 0.039853013046*t1834;
  t2003 = t1981 + t1990 + t1995;
  t1948 = -1.*t1946*t1739;
  t1978 = -1.*t1974*t1508;
  t2008 = -1.*t2003*t1609;
  t2023 = 0. + t428 + t429 + t920 + t437 + t962 + t1002 + t1558 + t1561 + t1562 + t1948 + t1978 + t2008;
  t1889 = -0.8656776547239999*t1855;
  t1890 = 1. + t1889;
  t2051 = t1946*t1651;
  t2059 = t1974*t1583;
  t2060 = t2003*t1618;
  t2063 = 0. + var1[1] + t1054 + t403 + t418 + t1055 + t1057 + t1525 + t1537 + t1554 + t2051 + t2059 + t2060;
  t2024 = t1834*t1651;
  t2025 = -0.340999127418*t1855*t1583;
  t2026 = t1861*t1618;
  t2029 = t2024 + t2025 + t2026;
  t2073 = t1886*t1651;
  t2081 = t1890*t1583;
  t2082 = -0.340999127418*t1855*t1618;
  t2085 = t2073 + t2081 + t2082;
  t1903 = -1.000000637725*t1855;
  t1909 = 1. + t1903;
  t1835 = t1834*t1426;
  t1858 = -0.340999127418*t1855*t1448;
  t1864 = t1861*t1490;
  t1877 = t1835 + t1858 + t1864;
  t2040 = t1834*t1739;
  t2041 = -0.340999127418*t1855*t1508;
  t2045 = t1861*t1609;
  t2050 = t2040 + t2041 + t2045;
  t2151 = t1946*t1739;
  t2155 = t1974*t1508;
  t2157 = t2003*t1609;
  t2158 = 0. + var1[2] + t550 + t1172 + t551 + t1174 + t1177 + t1682 + t1686 + t1687 + t2151 + t2155 + t2157;
  t1887 = t1886*t1426;
  t1895 = t1890*t1448;
  t1897 = -0.340999127418*t1855*t1490;
  t1899 = t1887 + t1895 + t1897;
  t2097 = t1886*t1739;
  t2101 = t1890*t1508;
  t2110 = -0.340999127418*t1855*t1609;
  t2113 = t2097 + t2101 + t2110;
  t2162 = -1.*t1946*t1426;
  t2164 = -1.*t1974*t1448;
  t2175 = -1.*t2003*t1490;
  t2176 = 0. + t37 + t1212 + t538 + t540 + t1219 + t1221 + t1666 + t1667 + t1668 + t2162 + t2164 + t2175;
  t2230 = t1946*t1426;
  t2232 = t1974*t1448;
  t2233 = t2003*t1490;
  t2235 = 0. + var1[0] + t1293 + t608 + t616 + t1297 + t1299 + t1771 + t1776 + t1777 + t2230 + t2232 + t2233;
  t2247 = -1.*t1946*t1651;
  t2249 = -1.*t1974*t1583;
  t2251 = -1.*t2003*t1618;
  t2273 = 0. + t584 + t1313 + t586 + t590 + t1314 + t1319 + t1749 + t1752 + t1758 + t2247 + t2249 + t2251;
  t2319 = Sin(var1[21]);
  t1910 = t1909*t1426;
  t1923 = t1918*t1448;
  t1930 = t1927*t1490;
  t1931 = t1910 + t1923 + t1930;
  t2329 = Cos(var1[21]);
  t2331 = -1.*t2329;
  t2333 = 1. + t2331;
  t2365 = -0.134322983001*t2333;
  t2370 = 1. + t2365;
  t2382 = 0.366501*t2319;
  t2383 = 0. + t2382;
  t2222 = t1909*t1739;
  t2223 = t1918*t1508;
  t2224 = t1927*t1609;
  t2225 = t2222 + t2223 + t2224;
  t2424 = 0.930418*t2319;
  t2428 = 0. + t2424;
  t2324 = -0.930418*t2319;
  t2326 = 0. + t2324;
  t2320 = -0.366501*t2319;
  t2321 = 0. + t2320;
  t2135 = t1909*t1651;
  t2136 = t1918*t1583;
  t2138 = t1927*t1618;
  t2140 = t2135 + t2136 + t2138;
  t2451 = 5.7930615939377813e-8*var1[21];
  t2461 = -0.23261833304643187*t2333;
  t2463 = -0.262809976934*t2383;
  t2464 = t2451 + t2461 + t2463;
  t2476 = -2.281945176511838e-8*var1[21];
  t2481 = -0.5905366811997648*t2333;
  t2486 = -0.262809976934*t2428;
  t2487 = t2476 + t2481 + t2486;
  t2490 = -0.26281014453449253*t2333;
  t2492 = -0.5905363046000001*t2326;
  t2496 = -0.23261818470000004*t2321;
  t2497 = 0. + t2490 + t2492 + t2496;
  t2419 = -0.8656776547239999*t2333;
  t2420 = 1. + t2419;
  t2475 = t2464*t2029;
  t2488 = t2487*t2085;
  t2498 = t2497*t2140;
  t2500 = 0. + var1[1] + t1054 + t403 + t418 + t1055 + t1057 + t1525 + t1537 + t1554 + t2051 + t2059 + t2060 + t2475 + t2488 + t2498;
  t2503 = -1.*t2464*t2050;
  t2505 = -1.*t2487*t2113;
  t2507 = -1.*t2497*t2225;
  t2512 = 0. + t428 + t429 + t920 + t437 + t962 + t1002 + t1558 + t1561 + t1562 + t1948 + t1978 + t2008 + t2503 + t2505 + t2507;
  t2334 = -1.000000637725*t2333;
  t2336 = 1. + t2334;
  t2516 = t2370*t2029;
  t2518 = -0.340999127418*t2333*t2085;
  t2522 = t2383*t2140;
  t2526 = t2516 + t2518 + t2522;
  t2555 = -0.340999127418*t2333*t2029;
  t2556 = t2420*t2085;
  t2558 = t2428*t2140;
  t2560 = t2555 + t2556 + t2558;
  t2440 = t2370*t2050;
  t2448 = -0.340999127418*t2333*t2113;
  t2449 = t2383*t2225;
  t2450 = t2440 + t2448 + t2449;
  t2371 = t2370*t1877;
  t2376 = -0.340999127418*t2333*t1899;
  t2387 = t2383*t1931;
  t2391 = t2371 + t2376 + t2387;
  t2546 = -0.340999127418*t2333*t2050;
  t2547 = t2420*t2113;
  t2548 = t2428*t2225;
  t2549 = t2546 + t2547 + t2548;
  t2589 = -1.*t2464*t1877;
  t2591 = -1.*t2487*t1899;
  t2593 = -1.*t2497*t1931;
  t2595 = 0. + t37 + t1212 + t538 + t540 + t1219 + t1221 + t1666 + t1667 + t1668 + t2162 + t2164 + t2175 + t2589 + t2591 + t2593;
  t2604 = t2464*t2050;
  t2607 = t2487*t2113;
  t2610 = t2497*t2225;
  t2613 = 0. + var1[2] + t550 + t1172 + t551 + t1174 + t1177 + t1682 + t1686 + t1687 + t2151 + t2155 + t2157 + t2604 + t2607 + t2610;
  t2399 = -0.340999127418*t2333*t1877;
  t2422 = t2420*t1899;
  t2432 = t2428*t1931;
  t2433 = t2399 + t2422 + t2432;
  t2665 = -1.*t2464*t2029;
  t2667 = -1.*t2487*t2085;
  t2668 = -1.*t2497*t2140;
  t2672 = 0. + t584 + t1313 + t586 + t590 + t1314 + t1319 + t1749 + t1752 + t1758 + t2247 + t2249 + t2251 + t2665 + t2667 + t2668;
  t2679 = t2464*t1877;
  t2683 = t2487*t1899;
  t2685 = t2497*t1931;
  t2686 = 0. + var1[0] + t1293 + t608 + t616 + t1297 + t1299 + t1771 + t1776 + t1777 + t2230 + t2232 + t2233 + t2679 + t2683 + t2685;
  t2322 = t2321*t1877;
  t2328 = t2326*t1899;
  t2338 = t2336*t1931;
  t2344 = t2322 + t2328 + t2338;
  t2795 = Cos(var1[22]);
  t2799 = -1.*t2795;
  t2801 = 1. + t2799;
  t2776 = Sin(var1[22]);
  t2858 = -0.930418*t2776;
  t2859 = 0. + t2858;
  t2854 = -0.366501*t2776;
  t2856 = 0. + t2854;
  t2653 = t2321*t2050;
  t2654 = t2326*t2113;
  t2656 = t2336*t2225;
  t2659 = t2653 + t2654 + t2656;
  t2823 = 0.366501*t2776;
  t2827 = 0. + t2823;
  t2780 = 0.930418*t2776;
  t2783 = 0. + t2780;
  t2568 = t2321*t2029;
  t2573 = t2326*t2085;
  t2574 = t2336*t2140;
  t2575 = t2568 + t2573 + t2574;
  t2807 = -0.8656776547239999*t2801;
  t2812 = 1. + t2807;
  t2868 = 0.06199697675299678*t2801;
  t2877 = -0.823260828522*t2859;
  t2880 = -0.324290713329*t2856;
  t2881 = 0. + t2868 + t2877 + t2880;
  t2885 = 7.500378623168247e-8*var1[22];
  t2887 = -0.32429092013729516*t2801;
  t2888 = 0.061996937216*t2827;
  t2890 = t2885 + t2887 + t2888;
  t2894 = -2.95447451120871e-8*var1[22];
  t2896 = -0.8232613535360118*t2801;
  t2897 = 0.061996937216*t2783;
  t2900 = t2894 + t2896 + t2897;
  t2882 = -1.*t2881*t2659;
  t2891 = -1.*t2890*t2450;
  t2907 = -1.*t2900*t2549;
  t2911 = 0. + t428 + t429 + t920 + t437 + t962 + t1002 + t1558 + t1561 + t1562 + t1948 + t1978 + t2008 + t2503 + t2505 + t2507 + t2882 + t2891 + t2907;
  t2829 = -0.134322983001*t2801;
  t2837 = 1. + t2829;
  t2949 = t2881*t2575;
  t2950 = t2890*t2526;
  t2955 = t2900*t2560;
  t2958 = 0. + var1[1] + t1054 + t403 + t418 + t1055 + t1057 + t1525 + t1537 + t1554 + t2051 + t2059 + t2060 + t2475 + t2488 + t2498 + t2949 + t2950 + t2955;
  t2850 = -1.000000637725*t2801;
  t2851 = 1. + t2850;
  t2915 = t2783*t2575;
  t2917 = -0.340999127418*t2801*t2526;
  t2920 = t2812*t2560;
  t2927 = t2915 + t2917 + t2920;
  t2976 = t2827*t2575;
  t2980 = t2837*t2526;
  t2981 = -0.340999127418*t2801*t2560;
  t2982 = t2976 + t2980 + t2981;
  t3044 = t2851*t2575;
  t3047 = t2856*t2526;
  t3048 = t2859*t2560;
  t3050 = t3044 + t3047 + t3048;
  t2788 = t2783*t2344;
  t2803 = -0.340999127418*t2801*t2391;
  t2818 = t2812*t2433;
  t2820 = t2788 + t2803 + t2818;
  t2942 = t2783*t2659;
  t2945 = -0.340999127418*t2801*t2450;
  t2946 = t2812*t2549;
  t2947 = t2942 + t2945 + t2946;
  t3067 = t2881*t2659;
  t3068 = t2890*t2450;
  t3072 = t2900*t2549;
  t3078 = 0. + var1[2] + t550 + t1172 + t551 + t1174 + t1177 + t1682 + t1686 + t1687 + t2151 + t2155 + t2157 + t2604 + t2607 + t2610 + t3067 + t3068 + t3072;
  t2828 = t2827*t2344;
  t2838 = t2837*t2391;
  t2839 = -0.340999127418*t2801*t2433;
  t2841 = t2828 + t2838 + t2839;
  t2988 = t2827*t2659;
  t2991 = t2837*t2450;
  t2994 = -0.340999127418*t2801*t2549;
  t3006 = t2988 + t2991 + t2994;
  t3082 = -1.*t2881*t2344;
  t3098 = -1.*t2890*t2391;
  t3100 = -1.*t2900*t2433;
  t3104 = 0. + t37 + t1212 + t538 + t540 + t1219 + t1221 + t1666 + t1667 + t1668 + t2162 + t2164 + t2175 + t2589 + t2591 + t2593 + t3082 + t3098 + t3100;
  t3027 = t2851*t2659;
  t3033 = t2856*t2450;
  t3038 = t2859*t2549;
  t3039 = t3027 + t3033 + t3038;
  t2852 = t2851*t2344;
  t2857 = t2856*t2391;
  t2860 = t2859*t2433;
  t2863 = t2852 + t2857 + t2860;
  t3176 = t2881*t2344;
  t3177 = t2890*t2391;
  t3181 = t2900*t2433;
  t3184 = 0. + var1[0] + t1293 + t608 + t616 + t1297 + t1299 + t1771 + t1776 + t1777 + t2230 + t2232 + t2233 + t2679 + t2683 + t2685 + t3176 + t3177 + t3181;
  t3187 = -1.*t2881*t2575;
  t3188 = -1.*t2890*t2526;
  t3194 = -1.*t2900*t2560;
  t3197 = 0. + t584 + t1313 + t586 + t590 + t1314 + t1319 + t1749 + t1752 + t1758 + t2247 + t2249 + t2251 + t2665 + t2667 + t2668 + t3187 + t3188 + t3194;
  p_output1[0]=1.;
  p_output1[1]=0;
  p_output1[2]=0;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0;
  p_output1[7]=1.;
  p_output1[8]=0;
  p_output1[9]=0;
  p_output1[10]=0;
  p_output1[11]=0;
  p_output1[12]=0;
  p_output1[13]=0;
  p_output1[14]=1.;
  p_output1[15]=0;
  p_output1[16]=0;
  p_output1[17]=0;
  p_output1[18]=var1[1];
  p_output1[19]=t37;
  p_output1[20]=0;
  p_output1[21]=0;
  p_output1[22]=0;
  p_output1[23]=1.;
  p_output1[24]=-1.*t48*var1[2];
  p_output1[25]=-1.*t68*var1[2];
  p_output1[26]=t48*var1[0] + t68*var1[1];
  p_output1[27]=-1.*t68;
  p_output1[28]=t48;
  p_output1[29]=0;
  p_output1[30]=-1.*t96*var1[1] - 1.*t68*t86*var1[2];
  p_output1[31]=t96*var1[0] + t48*t86*var1[2];
  p_output1[32]=t68*t86*var1[0] - 1.*t48*t86*var1[1];
  p_output1[33]=t48*t86;
  p_output1[34]=t68*t86;
  p_output1[35]=-1.*t96;
  p_output1[36]=0;
  p_output1[37]=0;
  p_output1[38]=0;
  p_output1[39]=0;
  p_output1[40]=0;
  p_output1[41]=0;
  p_output1[42]=0;
  p_output1[43]=0;
  p_output1[44]=0;
  p_output1[45]=0;
  p_output1[46]=0;
  p_output1[47]=0;
  p_output1[48]=0;
  p_output1[49]=0;
  p_output1[50]=0;
  p_output1[51]=0;
  p_output1[52]=0;
  p_output1[53]=0;
  p_output1[54]=0;
  p_output1[55]=0;
  p_output1[56]=0;
  p_output1[57]=0;
  p_output1[58]=0;
  p_output1[59]=0;
  p_output1[60]=0;
  p_output1[61]=0;
  p_output1[62]=0;
  p_output1[63]=0;
  p_output1[64]=0;
  p_output1[65]=0;
  p_output1[66]=0;
  p_output1[67]=0;
  p_output1[68]=0;
  p_output1[69]=0;
  p_output1[70]=0;
  p_output1[71]=0;
  p_output1[72]=0;
  p_output1[73]=0;
  p_output1[74]=0;
  p_output1[75]=0;
  p_output1[76]=0;
  p_output1[77]=0;
  p_output1[78]=0;
  p_output1[79]=0;
  p_output1[80]=0;
  p_output1[81]=0;
  p_output1[82]=0;
  p_output1[83]=0;
  p_output1[84]=0;
  p_output1[85]=0;
  p_output1[86]=0;
  p_output1[87]=0;
  p_output1[88]=0;
  p_output1[89]=0;
  p_output1[90]=0;
  p_output1[91]=0;
  p_output1[92]=0;
  p_output1[93]=0;
  p_output1[94]=0;
  p_output1[95]=0;
  p_output1[96]=0;
  p_output1[97]=0;
  p_output1[98]=0;
  p_output1[99]=0;
  p_output1[100]=0;
  p_output1[101]=0;
  p_output1[102]=-0.091*t195 + t96*var1[1] + t68*t86*var1[2];
  p_output1[103]=-0.091*t219 - 1.*t96*var1[0] - 1.*t48*t86*var1[2];
  p_output1[104]=-0.091*t184*t86 - 1.*t68*t86*var1[0] + t48*t86*var1[1];
  p_output1[105]=0. - 1.*t48*t86;
  p_output1[106]=0. - 1.*t68*t86;
  p_output1[107]=0. + t96;
  p_output1[108]=-0.041869*t343 + 0.016493*t372 - 0.366501*(t399*t420 + t441*t461) - 0.930418*(t420*t482 + t441*t494) + 0.084668*t48*t86;
  p_output1[109]=-0.041869*t461 + 0.016493*t494 - 0.366501*(t399*t542 + t343*t552) - 0.930418*(t482*t542 + t372*t552) + 0.084668*t68*t86;
  p_output1[110]=-0.041869*t399 + 0.016493*t482 - 0.366501*(t343*t593 + t461*t629) - 0.930418*(t372*t593 + t494*t629) - 0.084668*t96;
  p_output1[111]=0. - 0.366501*t343 - 0.930418*t372;
  p_output1[112]=0. - 0.366501*t461 - 0.930418*t494;
  p_output1[113]=0. - 0.366501*t399 - 0.930418*t482;
  p_output1[114]=0.366501*(t1015*t1034 + t1053*t1058) - 0.930418*(t1015*t1095 + t1058*t1114) + 0.041869*t768 + 0.016493*t836 - 0.151852*t902;
  p_output1[115]=0.041869*t1034 + 0.016493*t1095 - 0.151852*t1154 + 0.366501*(t1053*t1224 + t1194*t768) - 0.930418*(t1114*t1224 + t1194*t836);
  p_output1[116]=0.041869*t1053 + 0.016493*t1114 - 0.151852*t1284 + 0.366501*(t1034*t1305 + t1327*t768) - 0.930418*(t1095*t1305 + t1327*t836);
  p_output1[117]=0. + 0.366501*t768 - 0.930418*t836;
  p_output1[118]=0. + 0.366501*t1034 - 0.930418*t1095;
  p_output1[119]=0. + 0.366501*t1053 - 0.930418*t1114;
  p_output1[120]=0.236705*t1426 + 0.03708*t1448 + 0.014606*t1490 - 0.366501*(t1508*t1556 + t1568*t1583) + 0.930418*(t1556*t1609 + t1568*t1618);
  p_output1[121]=0.03708*t1583 + 0.014606*t1618 + 0.236705*t1651 - 0.366501*(t1508*t1669 + t1448*t1689) + 0.930418*(t1609*t1669 + t1490*t1689);
  p_output1[122]=0.03708*t1508 + 0.014606*t1609 + 0.236705*t1739 - 0.366501*(t1448*t1763 + t1583*t1778) + 0.930418*(t1490*t1763 + t1618*t1778);
  p_output1[123]=0. - 0.366501*t1448 + 0.930418*t1490;
  p_output1[124]=0. - 0.366501*t1583 + 0.930418*t1618;
  p_output1[125]=0. - 0.366501*t1508 + 0.930418*t1609;
  p_output1[126]=-0.09632*t1877 - 0.244523*t1899 + 0.6347*t1931 + 0.930418*(t2023*t2029 + t2050*t2063) - 0.366501*(t2023*t2085 + t2063*t2113);
  p_output1[127]=-0.09632*t2029 - 0.244523*t2085 + 0.6347*t2140 + 0.930418*(t1877*t2158 + t2050*t2176) - 0.366501*(t1899*t2158 + t2113*t2176);
  p_output1[128]=-0.09632*t2050 - 0.244523*t2113 + 0.6347*t2225 + 0.930418*(t2029*t2235 + t1877*t2273) - 0.366501*(t2085*t2235 + t1899*t2273);
  p_output1[129]=0. + 0.930418*t1877 - 0.366501*t1899;
  p_output1[130]=0. + 0.930418*t2029 - 0.366501*t2085;
  p_output1[131]=0. + 0.930418*t2050 - 0.366501*t2113;
  p_output1[132]=0.884829*t2344 + 0.022722*t2391 + 0.057683*t2433 + 0.930418*(t2450*t2500 + t2512*t2526) - 0.366501*(t2500*t2549 + t2512*t2560);
  p_output1[133]=0.022722*t2526 + 0.057683*t2560 + 0.884829*t2575 + 0.930418*(t2450*t2595 + t2391*t2613) - 0.366501*(t2549*t2595 + t2433*t2613);
  p_output1[134]=0.022722*t2450 + 0.057683*t2549 + 0.884829*t2659 + 0.930418*(t2391*t2672 + t2526*t2686) - 0.366501*(t2433*t2672 + t2560*t2686);
  p_output1[135]=0. + 0.930418*t2391 - 0.366501*t2433;
  p_output1[136]=0. + 0.930418*t2526 - 0.366501*t2560;
  p_output1[137]=0. + 0.930418*t2450 - 0.366501*t2549;
  p_output1[138]=0.337139*t2820 - 0.671277*t2841 - 0.050068*t2863 + 0.553471*(t2911*t2927 + t2947*t2958) + 0.218018*(t2911*t2982 + t2958*t3006) + 0.803828*(t2958*t3039 + t2911*t3050);
  p_output1[139]=0.337139*t2927 - 0.671277*t2982 - 0.050068*t3050 + 0.553471*(t2820*t3078 + t2947*t3104) + 0.218018*(t2841*t3078 + t3006*t3104) + 0.803828*(t2863*t3078 + t3039*t3104);
  p_output1[140]=0.337139*t2947 - 0.671277*t3006 - 0.050068*t3039 + 0.553471*(t2927*t3184 + t2820*t3197) + 0.218018*(t2982*t3184 + t2841*t3197) + 0.803828*(t3050*t3184 + t2863*t3197);
  p_output1[141]=0. + 0.553471*t2820 + 0.218018*t2841 + 0.803828*t2863;
  p_output1[142]=0. + 0.553471*t2927 + 0.218018*t2982 + 0.803828*t3050;
  p_output1[143]=0. + 0.553471*t2947 + 0.218018*t3006 + 0.803828*t3039;
  p_output1[144]=0;
  p_output1[145]=0;
  p_output1[146]=0;
  p_output1[147]=0;
  p_output1[148]=0;
  p_output1[149]=0;
  p_output1[150]=0;
  p_output1[151]=0;
  p_output1[152]=0;
  p_output1[153]=0;
  p_output1[154]=0;
  p_output1[155]=0;
  p_output1[156]=0;
  p_output1[157]=0;
  p_output1[158]=0;
  p_output1[159]=0;
  p_output1[160]=0;
  p_output1[161]=0;
  p_output1[162]=0;
  p_output1[163]=0;
  p_output1[164]=0;
  p_output1[165]=0;
  p_output1[166]=0;
  p_output1[167]=0;
}



void Js_toe_roll_joint_right_src(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}
