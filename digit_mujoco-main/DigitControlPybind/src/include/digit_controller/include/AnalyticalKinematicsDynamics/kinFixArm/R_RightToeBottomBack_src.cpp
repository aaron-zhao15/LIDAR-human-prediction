/*
 * Automatically Generated from Mathematica.
 * Sun 16 Oct 2022 21:44:32 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "R_RightToeBottomBack_src.h"

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
  double t291;
  double t1089;
  double t1234;
  double t1208;
  double t1362;
  double t706;
  double t743;
  double t888;
  double t1417;
  double t1211;
  double t1370;
  double t1374;
  double t957;
  double t1418;
  double t1419;
  double t1422;
  double t230;
  double t296;
  double t297;
  double t1396;
  double t1460;
  double t1461;
  double t1543;
  double t1575;
  double t1576;
  double t1780;
  double t1781;
  double t1830;
  double t496;
  double t507;
  double t699;
  double t1494;
  double t1513;
  double t1527;
  double t1587;
  double t1593;
  double t1708;
  double t1709;
  double t1715;
  double t1716;
  double t1719;
  double t1721;
  double t1728;
  double t1740;
  double t1841;
  double t1867;
  double t1910;
  double t1923;
  double t1958;
  double t1963;
  double t1964;
  double t1966;
  double t1967;
  double t1969;
  double t2010;
  double t2013;
  double t2022;
  double t131;
  double t211;
  double t241;
  double t267;
  double t1700;
  double t1706;
  double t1707;
  double t1773;
  double t1836;
  double t1839;
  double t1972;
  double t1980;
  double t2023;
  double t2024;
  double t2031;
  double t2079;
  double t2093;
  double t2094;
  double t2097;
  double t2101;
  double t2110;
  double t2112;
  double t2115;
  double t2129;
  double t2132;
  double t2133;
  double t2144;
  double t2163;
  double t2276;
  double t2279;
  double t2282;
  double t221;
  double t228;
  double t2009;
  double t2104;
  double t2105;
  double t2109;
  double t2166;
  double t2168;
  double t2222;
  double t2225;
  double t2236;
  double t2239;
  double t2243;
  double t2254;
  double t2258;
  double t2261;
  double t2287;
  double t2290;
  double t2301;
  double t2306;
  double t2309;
  double t2310;
  double t2314;
  double t2317;
  double t2320;
  double t2322;
  double t2346;
  double t2351;
  double t2358;
  double t6;
  double t24;
  double t100;
  double t110;
  double t140;
  double t157;
  double t2191;
  double t2217;
  double t2218;
  double t2265;
  double t2283;
  double t2284;
  double t2324;
  double t2327;
  double t2371;
  double t2374;
  double t2380;
  double t2381;
  double t2387;
  double t2389;
  double t2391;
  double t2393;
  double t2404;
  double t2405;
  double t2407;
  double t2412;
  double t2413;
  double t2414;
  double t2416;
  double t2418;
  double t103;
  double t115;
  double t126;
  double t2329;
  double t2394;
  double t2398;
  double t2403;
  double t2419;
  double t2423;
  double t102;
  double t2451;
  double t2454;
  double t2460;
  double t2461;
  double t2462;
  double t2463;
  double t2464;
  double t2465;
  double t2480;
  double t2485;
  double t2486;
  double t2488;
  double t2490;
  double t2492;
  double t2493;
  double t2494;
  double t2496;
  double t2501;
  double t104;
  double t107;
  double t2563;
  double t2565;
  double t2569;
  double t2572;
  double t2573;
  double t2574;
  double t2570;
  double t2575;
  double t2578;
  double t2580;
  double t2582;
  double t2583;
  double t2559;
  double t2579;
  double t2584;
  double t2588;
  double t2590;
  double t2592;
  double t2593;
  double t2595;
  double t2604;
  double t2606;
  double t2607;
  double t2608;
  double t2589;
  double t2599;
  double t2614;
  double t2617;
  double t2620;
  double t2623;
  double t2626;
  double t2633;
  double t2641;
  double t2642;
  double t2659;
  double t2661;
  double t2619;
  double t2636;
  double t2664;
  double t2665;
  double t2670;
  double t2671;
  double t2672;
  double t2676;
  double t2678;
  double t2680;
  double t2682;
  double t2683;
  double t2436;
  double t2443;
  double t2669;
  double t2677;
  double t2684;
  double t2685;
  double t2692;
  double t2693;
  double t2694;
  double t2698;
  double t2712;
  double t2713;
  double t2714;
  double t2727;
  double t2474;
  double t2477;
  double t2478;
  double t2512;
  double t2515;
  double t2689;
  double t2710;
  double t2728;
  double t2737;
  double t2520;
  double t2526;
  double t2744;
  double t2759;
  double t2775;
  double t2780;
  double t2535;
  double t2539;
  double t2540;
  double t2783;
  double t2784;
  double t2788;
  double t2792;
  double t2811;
  double t2812;
  double t2819;
  double t2822;
  double t2823;
  double t2826;
  double t2810;
  double t2820;
  double t2829;
  double t2830;
  double t2833;
  double t2834;
  double t2835;
  double t2838;
  double t2840;
  double t2841;
  double t2844;
  double t2846;
  double t2832;
  double t2839;
  double t2854;
  double t2861;
  double t2865;
  double t2867;
  double t2874;
  double t2882;
  double t2886;
  double t2893;
  double t2894;
  double t2900;
  double t2863;
  double t2883;
  double t2903;
  double t2904;
  double t2913;
  double t2914;
  double t2915;
  double t2916;
  double t2921;
  double t2926;
  double t2929;
  double t2930;
  double t2911;
  double t2920;
  double t2931;
  double t2932;
  double t2941;
  double t2942;
  double t2943;
  double t2944;
  double t2949;
  double t2950;
  double t2952;
  double t2953;
  double t2937;
  double t2945;
  double t2957;
  double t2960;
  double t2964;
  double t2966;
  double t2967;
  double t2969;
  double t2979;
  double t2984;
  double t2988;
  double t2995;
  double t2430;
  double t2472;
  double t2506;
  double t2507;
  double t2518;
  double t2534;
  double t2544;
  double t2549;
  double t3029;
  double t3031;
  double t3034;
  double t3038;
  double t3041;
  double t3043;
  double t2743;
  double t2782;
  double t2793;
  double t2794;
  double t2801;
  double t2802;
  double t2803;
  double t2804;
  double t2963;
  double t2977;
  double t2997;
  double t3007;
  double t3009;
  double t3015;
  double t3019;
  double t3024;
  double t3032;
  double t3040;
  double t3047;
  double t3048;
  double t3070;
  double t3075;
  double t3080;
  double t3084;
  double t3090;
  double t3095;
  double t3096;
  double t3097;
  t291 = Cos(var1[3]);
  t1089 = Cos(var1[5]);
  t1234 = Sin(var1[3]);
  t1208 = Sin(var1[4]);
  t1362 = Sin(var1[5]);
  t706 = Cos(var1[14]);
  t743 = -1.*t706;
  t888 = 1. + t743;
  t1417 = Cos(var1[13]);
  t1211 = t291*t1089*t1208;
  t1370 = t1234*t1362;
  t1374 = t1211 + t1370;
  t957 = Sin(var1[13]);
  t1418 = -1.*t1089*t1234;
  t1419 = t291*t1208*t1362;
  t1422 = t1418 + t1419;
  t230 = Sin(var1[15]);
  t296 = Cos(var1[4]);
  t297 = Sin(var1[14]);
  t1396 = -1.*t957*t1374;
  t1460 = t1417*t1422;
  t1461 = t1396 + t1460;
  t1543 = t1417*t1374;
  t1575 = t957*t1422;
  t1576 = t1543 + t1575;
  t1780 = Cos(var1[15]);
  t1781 = -1.*t1780;
  t1830 = 1. + t1781;
  t496 = -0.366501*t297;
  t507 = 0. + t496;
  t699 = t291*t296*t507;
  t1494 = 0.340999127418*t888*t1461;
  t1513 = -0.134322983001*t888;
  t1527 = 1. + t1513;
  t1587 = t1527*t1576;
  t1593 = t699 + t1494 + t1587;
  t1708 = 0.930418*t297;
  t1709 = 0. + t1708;
  t1715 = t291*t296*t1709;
  t1716 = -0.8656776547239999*t888;
  t1719 = 1. + t1716;
  t1721 = t1719*t1461;
  t1728 = 0.340999127418*t888*t1576;
  t1740 = t1715 + t1721 + t1728;
  t1841 = -1.000000637725*t888;
  t1867 = 1. + t1841;
  t1910 = t1867*t291*t296;
  t1923 = -0.930418*t297;
  t1958 = 0. + t1923;
  t1963 = t1958*t1461;
  t1964 = 0.366501*t297;
  t1966 = 0. + t1964;
  t1967 = t1966*t1576;
  t1969 = t1910 + t1963 + t1967;
  t2010 = Cos(var1[16]);
  t2013 = -1.*t2010;
  t2022 = 1. + t2013;
  t131 = Sin(var1[17]);
  t211 = Sin(var1[16]);
  t241 = 0.930418*t230;
  t267 = 0. + t241;
  t1700 = t267*t1593;
  t1706 = 0.366501*t230;
  t1707 = 0. + t1706;
  t1773 = t1707*t1740;
  t1836 = -1.000000637725*t1830;
  t1839 = 1. + t1836;
  t1972 = t1839*t1969;
  t1980 = t1700 + t1773 + t1972;
  t2023 = -0.8656776547239999*t1830;
  t2024 = 1. + t2023;
  t2031 = t2024*t1593;
  t2079 = -0.340999127418*t1830*t1740;
  t2093 = -0.930418*t230;
  t2094 = 0. + t2093;
  t2097 = t2094*t1969;
  t2101 = t2031 + t2079 + t2097;
  t2110 = -0.340999127418*t1830*t1593;
  t2112 = -0.134322983001*t1830;
  t2115 = 1. + t2112;
  t2129 = t2115*t1740;
  t2132 = -0.366501*t230;
  t2133 = 0. + t2132;
  t2144 = t2133*t1969;
  t2163 = t2110 + t2129 + t2144;
  t2276 = Cos(var1[17]);
  t2279 = -1.*t2276;
  t2282 = 1. + t2279;
  t221 = 0.366501*t211;
  t228 = 0. + t221;
  t2009 = t228*t1980;
  t2104 = -0.340999127418*t2022*t2101;
  t2105 = -0.134322983001*t2022;
  t2109 = 1. + t2105;
  t2166 = t2109*t2163;
  t2168 = t2009 + t2104 + t2166;
  t2222 = 0.930418*t211;
  t2225 = 0. + t2222;
  t2236 = t2225*t1980;
  t2239 = -0.8656776547239999*t2022;
  t2243 = 1. + t2239;
  t2254 = t2243*t2101;
  t2258 = -0.340999127418*t2022*t2163;
  t2261 = t2236 + t2254 + t2258;
  t2287 = -1.000000637725*t2022;
  t2290 = 1. + t2287;
  t2301 = t2290*t1980;
  t2306 = -0.930418*t211;
  t2309 = 0. + t2306;
  t2310 = t2309*t2101;
  t2314 = -0.366501*t211;
  t2317 = 0. + t2314;
  t2320 = t2317*t2163;
  t2322 = t2301 + t2310 + t2320;
  t2346 = Cos(var1[18]);
  t2351 = -1.*t2346;
  t2358 = 1. + t2351;
  t6 = Cos(var1[19]);
  t24 = -1.*t6;
  t100 = 1. + t24;
  t110 = Sin(var1[18]);
  t140 = -0.366501*t131;
  t157 = 0. + t140;
  t2191 = t157*t2168;
  t2217 = -0.930418*t131;
  t2218 = 0. + t2217;
  t2265 = t2218*t2261;
  t2283 = -1.000000637725*t2282;
  t2284 = 1. + t2283;
  t2324 = t2284*t2322;
  t2327 = t2191 + t2265 + t2324;
  t2371 = -0.134322983001*t2282;
  t2374 = 1. + t2371;
  t2380 = t2374*t2168;
  t2381 = -0.340999127418*t2282*t2261;
  t2387 = 0.366501*t131;
  t2389 = 0. + t2387;
  t2391 = t2389*t2322;
  t2393 = t2380 + t2381 + t2391;
  t2404 = -0.340999127418*t2282*t2168;
  t2405 = -0.8656776547239999*t2282;
  t2407 = 1. + t2405;
  t2412 = t2407*t2261;
  t2413 = 0.930418*t131;
  t2414 = 0. + t2413;
  t2416 = t2414*t2322;
  t2418 = t2404 + t2412 + t2416;
  t103 = Sin(var1[19]);
  t115 = 0.930418*t110;
  t126 = 0. + t115;
  t2329 = t126*t2327;
  t2394 = -0.340999127418*t2358*t2393;
  t2398 = -0.8656776547239999*t2358;
  t2403 = 1. + t2398;
  t2419 = t2403*t2418;
  t2423 = t2329 + t2394 + t2419;
  t102 = 0.120666640478*t100;
  t2451 = 0.366501*t110;
  t2454 = 0. + t2451;
  t2460 = t2454*t2327;
  t2461 = -0.134322983001*t2358;
  t2462 = 1. + t2461;
  t2463 = t2462*t2393;
  t2464 = -0.340999127418*t2358*t2418;
  t2465 = t2460 + t2463 + t2464;
  t2480 = -1.000000637725*t2358;
  t2485 = 1. + t2480;
  t2486 = t2485*t2327;
  t2488 = -0.366501*t110;
  t2490 = 0. + t2488;
  t2492 = t2490*t2393;
  t2493 = -0.930418*t110;
  t2494 = 0. + t2493;
  t2496 = t2494*t2418;
  t2501 = t2486 + t2492 + t2496;
  t104 = 0.803828*t103;
  t107 = t102 + t104;
  t2563 = t1089*t1234*t1208;
  t2565 = -1.*t291*t1362;
  t2569 = t2563 + t2565;
  t2572 = t291*t1089;
  t2573 = t1234*t1208*t1362;
  t2574 = t2572 + t2573;
  t2570 = -1.*t957*t2569;
  t2575 = t1417*t2574;
  t2578 = t2570 + t2575;
  t2580 = t1417*t2569;
  t2582 = t957*t2574;
  t2583 = t2580 + t2582;
  t2559 = t296*t507*t1234;
  t2579 = 0.340999127418*t888*t2578;
  t2584 = t1527*t2583;
  t2588 = t2559 + t2579 + t2584;
  t2590 = t296*t1709*t1234;
  t2592 = t1719*t2578;
  t2593 = 0.340999127418*t888*t2583;
  t2595 = t2590 + t2592 + t2593;
  t2604 = t1867*t296*t1234;
  t2606 = t1958*t2578;
  t2607 = t1966*t2583;
  t2608 = t2604 + t2606 + t2607;
  t2589 = t267*t2588;
  t2599 = t1707*t2595;
  t2614 = t1839*t2608;
  t2617 = t2589 + t2599 + t2614;
  t2620 = t2024*t2588;
  t2623 = -0.340999127418*t1830*t2595;
  t2626 = t2094*t2608;
  t2633 = t2620 + t2623 + t2626;
  t2641 = -0.340999127418*t1830*t2588;
  t2642 = t2115*t2595;
  t2659 = t2133*t2608;
  t2661 = t2641 + t2642 + t2659;
  t2619 = t228*t2617;
  t2636 = -0.340999127418*t2022*t2633;
  t2664 = t2109*t2661;
  t2665 = t2619 + t2636 + t2664;
  t2670 = t2225*t2617;
  t2671 = t2243*t2633;
  t2672 = -0.340999127418*t2022*t2661;
  t2676 = t2670 + t2671 + t2672;
  t2678 = t2290*t2617;
  t2680 = t2309*t2633;
  t2682 = t2317*t2661;
  t2683 = t2678 + t2680 + t2682;
  t2436 = -0.952469601425*t100;
  t2443 = 1. + t2436;
  t2669 = t157*t2665;
  t2677 = t2218*t2676;
  t2684 = t2284*t2683;
  t2685 = t2669 + t2677 + t2684;
  t2692 = t2374*t2665;
  t2693 = -0.340999127418*t2282*t2676;
  t2694 = t2389*t2683;
  t2698 = t2692 + t2693 + t2694;
  t2712 = -0.340999127418*t2282*t2665;
  t2713 = t2407*t2676;
  t2714 = t2414*t2683;
  t2727 = t2712 + t2713 + t2714;
  t2474 = 0.175248972904*t100;
  t2477 = -0.553471*t103;
  t2478 = t2474 + t2477;
  t2512 = -0.693671301908*t100;
  t2515 = 1. + t2512;
  t2689 = t126*t2685;
  t2710 = -0.340999127418*t2358*t2698;
  t2728 = t2403*t2727;
  t2737 = t2689 + t2710 + t2728;
  t2520 = -0.803828*t103;
  t2526 = t102 + t2520;
  t2744 = t2454*t2685;
  t2759 = t2462*t2698;
  t2775 = -0.340999127418*t2358*t2727;
  t2780 = t2744 + t2759 + t2775;
  t2535 = 0.444895486988*t100;
  t2539 = 0.218018*t103;
  t2540 = t2535 + t2539;
  t2783 = t2485*t2685;
  t2784 = t2490*t2698;
  t2788 = t2494*t2727;
  t2792 = t2783 + t2784 + t2788;
  t2811 = -1.*t296*t1089*t957;
  t2812 = t1417*t296*t1362;
  t2819 = t2811 + t2812;
  t2822 = t1417*t296*t1089;
  t2823 = t296*t957*t1362;
  t2826 = t2822 + t2823;
  t2810 = -1.*t507*t1208;
  t2820 = 0.340999127418*t888*t2819;
  t2829 = t1527*t2826;
  t2830 = t2810 + t2820 + t2829;
  t2833 = -1.*t1709*t1208;
  t2834 = t1719*t2819;
  t2835 = 0.340999127418*t888*t2826;
  t2838 = t2833 + t2834 + t2835;
  t2840 = -1.*t1867*t1208;
  t2841 = t1958*t2819;
  t2844 = t1966*t2826;
  t2846 = t2840 + t2841 + t2844;
  t2832 = t267*t2830;
  t2839 = t1707*t2838;
  t2854 = t1839*t2846;
  t2861 = t2832 + t2839 + t2854;
  t2865 = t2024*t2830;
  t2867 = -0.340999127418*t1830*t2838;
  t2874 = t2094*t2846;
  t2882 = t2865 + t2867 + t2874;
  t2886 = -0.340999127418*t1830*t2830;
  t2893 = t2115*t2838;
  t2894 = t2133*t2846;
  t2900 = t2886 + t2893 + t2894;
  t2863 = t228*t2861;
  t2883 = -0.340999127418*t2022*t2882;
  t2903 = t2109*t2900;
  t2904 = t2863 + t2883 + t2903;
  t2913 = t2225*t2861;
  t2914 = t2243*t2882;
  t2915 = -0.340999127418*t2022*t2900;
  t2916 = t2913 + t2914 + t2915;
  t2921 = t2290*t2861;
  t2926 = t2309*t2882;
  t2929 = t2317*t2900;
  t2930 = t2921 + t2926 + t2929;
  t2911 = t157*t2904;
  t2920 = t2218*t2916;
  t2931 = t2284*t2930;
  t2932 = t2911 + t2920 + t2931;
  t2941 = t2374*t2904;
  t2942 = -0.340999127418*t2282*t2916;
  t2943 = t2389*t2930;
  t2944 = t2941 + t2942 + t2943;
  t2949 = -0.340999127418*t2282*t2904;
  t2950 = t2407*t2916;
  t2952 = t2414*t2930;
  t2953 = t2949 + t2950 + t2952;
  t2937 = t126*t2932;
  t2945 = -0.340999127418*t2358*t2944;
  t2957 = t2403*t2953;
  t2960 = t2937 + t2945 + t2957;
  t2964 = t2454*t2932;
  t2966 = t2462*t2944;
  t2967 = -0.340999127418*t2358*t2953;
  t2969 = t2964 + t2966 + t2967;
  t2979 = t2485*t2932;
  t2984 = t2490*t2944;
  t2988 = t2494*t2953;
  t2995 = t2979 + t2984 + t2988;
  t2430 = t107*t2423;
  t2472 = t2443*t2465;
  t2506 = t2478*t2501;
  t2507 = t2430 + t2472 + t2506;
  t2518 = t2515*t2423;
  t2534 = t2526*t2465;
  t2544 = t2540*t2501;
  t2549 = t2518 + t2534 + t2544;
  t3029 = -0.218018*t103;
  t3031 = t2535 + t3029;
  t3034 = 0.553471*t103;
  t3038 = t2474 + t3034;
  t3041 = -0.353861996165*t100;
  t3043 = 1. + t3041;
  t2743 = t107*t2737;
  t2782 = t2443*t2780;
  t2793 = t2478*t2792;
  t2794 = t2743 + t2782 + t2793;
  t2801 = t2515*t2737;
  t2802 = t2526*t2780;
  t2803 = t2540*t2792;
  t2804 = t2801 + t2802 + t2803;
  t2963 = t107*t2960;
  t2977 = t2443*t2969;
  t2997 = t2478*t2995;
  t3007 = t2963 + t2977 + t2997;
  t3009 = t2515*t2960;
  t3015 = t2526*t2969;
  t3019 = t2540*t2995;
  t3024 = t3009 + t3015 + t3019;
  t3032 = t3031*t2423;
  t3040 = t3038*t2465;
  t3047 = t3043*t2501;
  t3048 = t3032 + t3040 + t3047;
  t3070 = t3031*t2737;
  t3075 = t3038*t2780;
  t3080 = t3043*t2792;
  t3084 = t3070 + t3075 + t3080;
  t3090 = t3031*t2960;
  t3095 = t3038*t2969;
  t3096 = t3043*t2995;
  t3097 = t3090 + t3095 + t3096;
  p_output1[0]=-0.930418*t2507 + 0.366501*t2549;
  p_output1[1]=-0.930418*t2794 + 0.366501*t2804;
  p_output1[2]=-0.930418*t3007 + 0.366501*t3024;
  p_output1[3]=-0.294604*t2507 - 0.747896*t2549 + 0.594863*t3048;
  p_output1[4]=-0.294604*t2794 - 0.747896*t2804 + 0.594863*t3084;
  p_output1[5]=-0.294604*t3007 - 0.747896*t3024 + 0.594863*t3097;
  p_output1[6]=0.218018*t2507 + 0.553471*t2549 + 0.803828*t3048;
  p_output1[7]=0.218018*t2794 + 0.553471*t2804 + 0.803828*t3084;
  p_output1[8]=0.218018*t3007 + 0.553471*t3024 + 0.803828*t3097;
}



void R_RightToeBottomBack_src(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}
