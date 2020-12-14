/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*               This file is part of the program and library                */
/*    PaPILO --- Parallel Presolve for Integer and Linear Optimization       */
/*                                                                           */
/* Copyright (C) 2020  Konrad-Zuse-Zentrum                                   */
/*                     fuer Informationstechnik Berlin                       */
/*                                                                           */
/* This program is free software: you can redistribute it and/or modify      */
/* it under the terms of the GNU Lesser General Public License as published  */
/* by the Free Software Foundation, either version 3 of the License, or      */
/* (at your option) any later version.                                       */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.    */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef PAPILO_TEST_INSTANCES_RGN
#define PAPILO_TEST_INSTANCES_RGN

#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

namespace papilo
{
namespace instances
{

Problem<double>
rgn()
{
   /// PROBLEM BUILDER CODE
   Vec<double> coeffobj{
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
   };
   Vec<double> lbs{
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   };
   Vec<uint8_t> lbInf{
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<double> ubs{
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,
       100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
       100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
       2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,
       2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,
       2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,
       2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,
       100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
       100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
   };
   Vec<uint8_t> ubInf{
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<uint8_t> isIntegral{
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<uint8_t> lhsIsInf{
       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<double> lhs{
       0.0,  0.0,  0.0,  0.0,  -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5,
       -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5,
   };
   Vec<uint8_t> rhsIsInf{
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<double> rhs{
       1.0,  1.0,  1.0,  1.0,  -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5,
       -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5,
   };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 0, 3, 1.0 },
       std::tuple<int, int, double>{ 0, 4, 1.0 },
       std::tuple<int, int, double>{ 0, 5, 1.0 },
       std::tuple<int, int, double>{ 0, 6, 1.0 },
       std::tuple<int, int, double>{ 0, 7, 1.0 },
       std::tuple<int, int, double>{ 0, 8, 1.0 },
       std::tuple<int, int, double>{ 0, 9, 1.0 },
       std::tuple<int, int, double>{ 0, 10, 1.0 },
       std::tuple<int, int, double>{ 0, 11, 1.0 },
       std::tuple<int, int, double>{ 0, 12, 1.0 },
       std::tuple<int, int, double>{ 0, 13, 1.0 },
       std::tuple<int, int, double>{ 0, 14, 1.0 },
       std::tuple<int, int, double>{ 0, 15, 1.0 },
       std::tuple<int, int, double>{ 0, 16, 1.0 },
       std::tuple<int, int, double>{ 0, 17, 1.0 },
       std::tuple<int, int, double>{ 0, 18, 1.0 },
       std::tuple<int, int, double>{ 0, 19, 1.0 },
       std::tuple<int, int, double>{ 0, 20, 1.0 },
       std::tuple<int, int, double>{ 0, 21, 1.0 },
       std::tuple<int, int, double>{ 0, 22, 1.0 },
       std::tuple<int, int, double>{ 0, 23, 1.0 },
       std::tuple<int, int, double>{ 0, 24, 1.0 },
       std::tuple<int, int, double>{ 1, 25, 1.0 },
       std::tuple<int, int, double>{ 1, 26, 1.0 },
       std::tuple<int, int, double>{ 1, 27, 1.0 },
       std::tuple<int, int, double>{ 1, 28, 1.0 },
       std::tuple<int, int, double>{ 1, 29, 1.0 },
       std::tuple<int, int, double>{ 1, 30, 1.0 },
       std::tuple<int, int, double>{ 1, 31, 1.0 },
       std::tuple<int, int, double>{ 1, 32, 1.0 },
       std::tuple<int, int, double>{ 1, 33, 1.0 },
       std::tuple<int, int, double>{ 1, 34, 1.0 },
       std::tuple<int, int, double>{ 1, 35, 1.0 },
       std::tuple<int, int, double>{ 1, 36, 1.0 },
       std::tuple<int, int, double>{ 1, 37, 1.0 },
       std::tuple<int, int, double>{ 1, 38, 1.0 },
       std::tuple<int, int, double>{ 1, 39, 1.0 },
       std::tuple<int, int, double>{ 1, 40, 1.0 },
       std::tuple<int, int, double>{ 1, 41, 1.0 },
       std::tuple<int, int, double>{ 1, 42, 1.0 },
       std::tuple<int, int, double>{ 1, 43, 1.0 },
       std::tuple<int, int, double>{ 1, 44, 1.0 },
       std::tuple<int, int, double>{ 1, 45, 1.0 },
       std::tuple<int, int, double>{ 1, 46, 1.0 },
       std::tuple<int, int, double>{ 1, 47, 1.0 },
       std::tuple<int, int, double>{ 1, 48, 1.0 },
       std::tuple<int, int, double>{ 1, 49, 1.0 },
       std::tuple<int, int, double>{ 2, 50, 1.0 },
       std::tuple<int, int, double>{ 2, 51, 1.0 },
       std::tuple<int, int, double>{ 2, 52, 1.0 },
       std::tuple<int, int, double>{ 2, 53, 1.0 },
       std::tuple<int, int, double>{ 2, 54, 1.0 },
       std::tuple<int, int, double>{ 2, 55, 1.0 },
       std::tuple<int, int, double>{ 2, 56, 1.0 },
       std::tuple<int, int, double>{ 2, 57, 1.0 },
       std::tuple<int, int, double>{ 2, 58, 1.0 },
       std::tuple<int, int, double>{ 2, 59, 1.0 },
       std::tuple<int, int, double>{ 2, 60, 1.0 },
       std::tuple<int, int, double>{ 2, 61, 1.0 },
       std::tuple<int, int, double>{ 2, 62, 1.0 },
       std::tuple<int, int, double>{ 2, 63, 1.0 },
       std::tuple<int, int, double>{ 2, 64, 1.0 },
       std::tuple<int, int, double>{ 2, 65, 1.0 },
       std::tuple<int, int, double>{ 2, 66, 1.0 },
       std::tuple<int, int, double>{ 2, 67, 1.0 },
       std::tuple<int, int, double>{ 2, 68, 1.0 },
       std::tuple<int, int, double>{ 2, 69, 1.0 },
       std::tuple<int, int, double>{ 2, 70, 1.0 },
       std::tuple<int, int, double>{ 2, 71, 1.0 },
       std::tuple<int, int, double>{ 2, 72, 1.0 },
       std::tuple<int, int, double>{ 2, 73, 1.0 },
       std::tuple<int, int, double>{ 2, 74, 1.0 },
       std::tuple<int, int, double>{ 3, 75, 1.0 },
       std::tuple<int, int, double>{ 3, 76, 1.0 },
       std::tuple<int, int, double>{ 3, 77, 1.0 },
       std::tuple<int, int, double>{ 3, 78, 1.0 },
       std::tuple<int, int, double>{ 3, 79, 1.0 },
       std::tuple<int, int, double>{ 3, 80, 1.0 },
       std::tuple<int, int, double>{ 3, 81, 1.0 },
       std::tuple<int, int, double>{ 3, 82, 1.0 },
       std::tuple<int, int, double>{ 3, 83, 1.0 },
       std::tuple<int, int, double>{ 3, 84, 1.0 },
       std::tuple<int, int, double>{ 3, 85, 1.0 },
       std::tuple<int, int, double>{ 3, 86, 1.0 },
       std::tuple<int, int, double>{ 3, 87, 1.0 },
       std::tuple<int, int, double>{ 3, 88, 1.0 },
       std::tuple<int, int, double>{ 3, 89, 1.0 },
       std::tuple<int, int, double>{ 3, 90, 1.0 },
       std::tuple<int, int, double>{ 3, 91, 1.0 },
       std::tuple<int, int, double>{ 3, 92, 1.0 },
       std::tuple<int, int, double>{ 3, 93, 1.0 },
       std::tuple<int, int, double>{ 3, 94, 1.0 },
       std::tuple<int, int, double>{ 3, 95, 1.0 },
       std::tuple<int, int, double>{ 3, 96, 1.0 },
       std::tuple<int, int, double>{ 3, 97, 1.0 },
       std::tuple<int, int, double>{ 3, 98, 1.0 },
       std::tuple<int, int, double>{ 3, 99, 1.0 },
       std::tuple<int, int, double>{ 4, 0, -4.60000002 },
       std::tuple<int, int, double>{ 4, 5, -4.60000002 },
       std::tuple<int, int, double>{ 4, 6, -4.60000002 },
       std::tuple<int, int, double>{ 4, 7, -4.60000002 },
       std::tuple<int, int, double>{ 4, 8, -4.60000002 },
       std::tuple<int, int, double>{ 4, 15, -4.60000002 },
       std::tuple<int, int, double>{ 4, 16, -4.60000002 },
       std::tuple<int, int, double>{ 4, 17, -4.60000002 },
       std::tuple<int, int, double>{ 4, 18, -4.60000002 },
       std::tuple<int, int, double>{ 4, 19, -4.60000002 },
       std::tuple<int, int, double>{ 4, 20, -4.60000002 },
       std::tuple<int, int, double>{ 4, 100, 1.0 },
       std::tuple<int, int, double>{ 4, 120, 1.0 },
       std::tuple<int, int, double>{ 4, 140, -1.0 },
       std::tuple<int, int, double>{ 4, 160, -1.0 },
       std::tuple<int, int, double>{ 5, 1, -4.60000002 },
       std::tuple<int, int, double>{ 5, 5, -4.60000002 },
       std::tuple<int, int, double>{ 5, 9, -4.60000002 },
       std::tuple<int, int, double>{ 5, 10, -4.60000002 },
       std::tuple<int, int, double>{ 5, 11, -4.60000002 },
       std::tuple<int, int, double>{ 5, 15, -4.60000002 },
       std::tuple<int, int, double>{ 5, 16, -4.60000002 },
       std::tuple<int, int, double>{ 5, 17, -4.60000002 },
       std::tuple<int, int, double>{ 5, 21, -4.60000002 },
       std::tuple<int, int, double>{ 5, 22, -4.60000002 },
       std::tuple<int, int, double>{ 5, 23, -4.60000002 },
       std::tuple<int, int, double>{ 5, 104, 1.0 },
       std::tuple<int, int, double>{ 5, 124, 1.0 },
       std::tuple<int, int, double>{ 5, 144, -1.0 },
       std::tuple<int, int, double>{ 5, 164, -1.0 },
       std::tuple<int, int, double>{ 6, 2, -4.60000002 },
       std::tuple<int, int, double>{ 6, 6, -4.60000002 },
       std::tuple<int, int, double>{ 6, 9, -4.60000002 },
       std::tuple<int, int, double>{ 6, 12, -4.60000002 },
       std::tuple<int, int, double>{ 6, 13, -4.60000002 },
       std::tuple<int, int, double>{ 6, 15, -4.60000002 },
       std::tuple<int, int, double>{ 6, 18, -4.60000002 },
       std::tuple<int, int, double>{ 6, 19, -4.60000002 },
       std::tuple<int, int, double>{ 6, 21, -4.60000002 },
       std::tuple<int, int, double>{ 6, 22, -4.60000002 },
       std::tuple<int, int, double>{ 6, 24, -4.60000002 },
       std::tuple<int, int, double>{ 6, 108, 1.0 },
       std::tuple<int, int, double>{ 6, 128, 1.0 },
       std::tuple<int, int, double>{ 6, 148, -1.0 },
       std::tuple<int, int, double>{ 6, 168, -1.0 },
       std::tuple<int, int, double>{ 7, 3, -4.60000002 },
       std::tuple<int, int, double>{ 7, 7, -4.60000002 },
       std::tuple<int, int, double>{ 7, 10, -4.60000002 },
       std::tuple<int, int, double>{ 7, 12, -4.60000002 },
       std::tuple<int, int, double>{ 7, 14, -4.60000002 },
       std::tuple<int, int, double>{ 7, 16, -4.60000002 },
       std::tuple<int, int, double>{ 7, 18, -4.60000002 },
       std::tuple<int, int, double>{ 7, 20, -4.60000002 },
       std::tuple<int, int, double>{ 7, 21, -4.60000002 },
       std::tuple<int, int, double>{ 7, 23, -4.60000002 },
       std::tuple<int, int, double>{ 7, 24, -4.60000002 },
       std::tuple<int, int, double>{ 7, 112, 1.0 },
       std::tuple<int, int, double>{ 7, 132, 1.0 },
       std::tuple<int, int, double>{ 7, 152, -1.0 },
       std::tuple<int, int, double>{ 7, 172, -1.0 },
       std::tuple<int, int, double>{ 8, 4, -4.60000002 },
       std::tuple<int, int, double>{ 8, 8, -4.60000002 },
       std::tuple<int, int, double>{ 8, 11, -4.60000002 },
       std::tuple<int, int, double>{ 8, 13, -4.60000002 },
       std::tuple<int, int, double>{ 8, 14, -4.60000002 },
       std::tuple<int, int, double>{ 8, 17, -4.60000002 },
       std::tuple<int, int, double>{ 8, 19, -4.60000002 },
       std::tuple<int, int, double>{ 8, 20, -4.60000002 },
       std::tuple<int, int, double>{ 8, 22, -4.60000002 },
       std::tuple<int, int, double>{ 8, 23, -4.60000002 },
       std::tuple<int, int, double>{ 8, 24, -4.60000002 },
       std::tuple<int, int, double>{ 8, 116, 1.0 },
       std::tuple<int, int, double>{ 8, 136, 1.0 },
       std::tuple<int, int, double>{ 8, 156, -1.0 },
       std::tuple<int, int, double>{ 8, 176, -1.0 },
       std::tuple<int, int, double>{ 9, 25, -4.60000002 },
       std::tuple<int, int, double>{ 9, 30, -4.60000002 },
       std::tuple<int, int, double>{ 9, 31, -4.60000002 },
       std::tuple<int, int, double>{ 9, 32, -4.60000002 },
       std::tuple<int, int, double>{ 9, 33, -4.60000002 },
       std::tuple<int, int, double>{ 9, 40, -4.60000002 },
       std::tuple<int, int, double>{ 9, 41, -4.60000002 },
       std::tuple<int, int, double>{ 9, 42, -4.60000002 },
       std::tuple<int, int, double>{ 9, 43, -4.60000002 },
       std::tuple<int, int, double>{ 9, 44, -4.60000002 },
       std::tuple<int, int, double>{ 9, 45, -4.60000002 },
       std::tuple<int, int, double>{ 9, 100, -1.0 },
       std::tuple<int, int, double>{ 9, 101, 1.0 },
       std::tuple<int, int, double>{ 9, 120, -1.0 },
       std::tuple<int, int, double>{ 9, 121, 1.0 },
       std::tuple<int, int, double>{ 9, 140, 1.0 },
       std::tuple<int, int, double>{ 9, 141, -1.0 },
       std::tuple<int, int, double>{ 9, 160, 1.0 },
       std::tuple<int, int, double>{ 9, 161, -1.0 },
       std::tuple<int, int, double>{ 10, 26, -4.60000002 },
       std::tuple<int, int, double>{ 10, 30, -4.60000002 },
       std::tuple<int, int, double>{ 10, 34, -4.60000002 },
       std::tuple<int, int, double>{ 10, 35, -4.60000002 },
       std::tuple<int, int, double>{ 10, 36, -4.60000002 },
       std::tuple<int, int, double>{ 10, 40, -4.60000002 },
       std::tuple<int, int, double>{ 10, 41, -4.60000002 },
       std::tuple<int, int, double>{ 10, 42, -4.60000002 },
       std::tuple<int, int, double>{ 10, 46, -4.60000002 },
       std::tuple<int, int, double>{ 10, 47, -4.60000002 },
       std::tuple<int, int, double>{ 10, 48, -4.60000002 },
       std::tuple<int, int, double>{ 10, 104, -1.0 },
       std::tuple<int, int, double>{ 10, 105, 1.0 },
       std::tuple<int, int, double>{ 10, 124, -1.0 },
       std::tuple<int, int, double>{ 10, 125, 1.0 },
       std::tuple<int, int, double>{ 10, 144, 1.0 },
       std::tuple<int, int, double>{ 10, 145, -1.0 },
       std::tuple<int, int, double>{ 10, 164, 1.0 },
       std::tuple<int, int, double>{ 10, 165, -1.0 },
       std::tuple<int, int, double>{ 11, 27, -4.60000002 },
       std::tuple<int, int, double>{ 11, 31, -4.60000002 },
       std::tuple<int, int, double>{ 11, 34, -4.60000002 },
       std::tuple<int, int, double>{ 11, 37, -4.60000002 },
       std::tuple<int, int, double>{ 11, 38, -4.60000002 },
       std::tuple<int, int, double>{ 11, 40, -4.60000002 },
       std::tuple<int, int, double>{ 11, 43, -4.60000002 },
       std::tuple<int, int, double>{ 11, 44, -4.60000002 },
       std::tuple<int, int, double>{ 11, 46, -4.60000002 },
       std::tuple<int, int, double>{ 11, 47, -4.60000002 },
       std::tuple<int, int, double>{ 11, 49, -4.60000002 },
       std::tuple<int, int, double>{ 11, 108, -1.0 },
       std::tuple<int, int, double>{ 11, 109, 1.0 },
       std::tuple<int, int, double>{ 11, 128, -1.0 },
       std::tuple<int, int, double>{ 11, 129, 1.0 },
       std::tuple<int, int, double>{ 11, 148, 1.0 },
       std::tuple<int, int, double>{ 11, 149, -1.0 },
       std::tuple<int, int, double>{ 11, 168, 1.0 },
       std::tuple<int, int, double>{ 11, 169, -1.0 },
       std::tuple<int, int, double>{ 12, 28, -4.60000002 },
       std::tuple<int, int, double>{ 12, 32, -4.60000002 },
       std::tuple<int, int, double>{ 12, 35, -4.60000002 },
       std::tuple<int, int, double>{ 12, 37, -4.60000002 },
       std::tuple<int, int, double>{ 12, 39, -4.60000002 },
       std::tuple<int, int, double>{ 12, 41, -4.60000002 },
       std::tuple<int, int, double>{ 12, 43, -4.60000002 },
       std::tuple<int, int, double>{ 12, 45, -4.60000002 },
       std::tuple<int, int, double>{ 12, 46, -4.60000002 },
       std::tuple<int, int, double>{ 12, 48, -4.60000002 },
       std::tuple<int, int, double>{ 12, 49, -4.60000002 },
       std::tuple<int, int, double>{ 12, 112, -1.0 },
       std::tuple<int, int, double>{ 12, 113, 1.0 },
       std::tuple<int, int, double>{ 12, 132, -1.0 },
       std::tuple<int, int, double>{ 12, 133, 1.0 },
       std::tuple<int, int, double>{ 12, 152, 1.0 },
       std::tuple<int, int, double>{ 12, 153, -1.0 },
       std::tuple<int, int, double>{ 12, 172, 1.0 },
       std::tuple<int, int, double>{ 12, 173, -1.0 },
       std::tuple<int, int, double>{ 13, 29, -4.60000002 },
       std::tuple<int, int, double>{ 13, 33, -4.60000002 },
       std::tuple<int, int, double>{ 13, 36, -4.60000002 },
       std::tuple<int, int, double>{ 13, 38, -4.60000002 },
       std::tuple<int, int, double>{ 13, 39, -4.60000002 },
       std::tuple<int, int, double>{ 13, 42, -4.60000002 },
       std::tuple<int, int, double>{ 13, 44, -4.60000002 },
       std::tuple<int, int, double>{ 13, 45, -4.60000002 },
       std::tuple<int, int, double>{ 13, 47, -4.60000002 },
       std::tuple<int, int, double>{ 13, 48, -4.60000002 },
       std::tuple<int, int, double>{ 13, 49, -4.60000002 },
       std::tuple<int, int, double>{ 13, 116, -1.0 },
       std::tuple<int, int, double>{ 13, 117, 1.0 },
       std::tuple<int, int, double>{ 13, 136, -1.0 },
       std::tuple<int, int, double>{ 13, 137, 1.0 },
       std::tuple<int, int, double>{ 13, 156, 1.0 },
       std::tuple<int, int, double>{ 13, 157, -1.0 },
       std::tuple<int, int, double>{ 13, 176, 1.0 },
       std::tuple<int, int, double>{ 13, 177, -1.0 },
       std::tuple<int, int, double>{ 14, 50, -4.60000002 },
       std::tuple<int, int, double>{ 14, 55, -4.60000002 },
       std::tuple<int, int, double>{ 14, 56, -4.60000002 },
       std::tuple<int, int, double>{ 14, 57, -4.60000002 },
       std::tuple<int, int, double>{ 14, 58, -4.60000002 },
       std::tuple<int, int, double>{ 14, 65, -4.60000002 },
       std::tuple<int, int, double>{ 14, 66, -4.60000002 },
       std::tuple<int, int, double>{ 14, 67, -4.60000002 },
       std::tuple<int, int, double>{ 14, 68, -4.60000002 },
       std::tuple<int, int, double>{ 14, 69, -4.60000002 },
       std::tuple<int, int, double>{ 14, 70, -4.60000002 },
       std::tuple<int, int, double>{ 14, 101, -1.0 },
       std::tuple<int, int, double>{ 14, 102, 1.0 },
       std::tuple<int, int, double>{ 14, 121, -1.0 },
       std::tuple<int, int, double>{ 14, 122, 1.0 },
       std::tuple<int, int, double>{ 14, 141, 1.0 },
       std::tuple<int, int, double>{ 14, 142, -1.0 },
       std::tuple<int, int, double>{ 14, 161, 1.0 },
       std::tuple<int, int, double>{ 14, 162, -1.0 },
       std::tuple<int, int, double>{ 15, 51, -4.60000002 },
       std::tuple<int, int, double>{ 15, 55, -4.60000002 },
       std::tuple<int, int, double>{ 15, 59, -4.60000002 },
       std::tuple<int, int, double>{ 15, 60, -4.60000002 },
       std::tuple<int, int, double>{ 15, 61, -4.60000002 },
       std::tuple<int, int, double>{ 15, 65, -4.60000002 },
       std::tuple<int, int, double>{ 15, 66, -4.60000002 },
       std::tuple<int, int, double>{ 15, 67, -4.60000002 },
       std::tuple<int, int, double>{ 15, 71, -4.60000002 },
       std::tuple<int, int, double>{ 15, 72, -4.60000002 },
       std::tuple<int, int, double>{ 15, 73, -4.60000002 },
       std::tuple<int, int, double>{ 15, 105, -1.0 },
       std::tuple<int, int, double>{ 15, 106, 1.0 },
       std::tuple<int, int, double>{ 15, 125, -1.0 },
       std::tuple<int, int, double>{ 15, 126, 1.0 },
       std::tuple<int, int, double>{ 15, 145, 1.0 },
       std::tuple<int, int, double>{ 15, 146, -1.0 },
       std::tuple<int, int, double>{ 15, 165, 1.0 },
       std::tuple<int, int, double>{ 15, 166, -1.0 },
       std::tuple<int, int, double>{ 16, 52, -4.60000002 },
       std::tuple<int, int, double>{ 16, 56, -4.60000002 },
       std::tuple<int, int, double>{ 16, 59, -4.60000002 },
       std::tuple<int, int, double>{ 16, 62, -4.60000002 },
       std::tuple<int, int, double>{ 16, 63, -4.60000002 },
       std::tuple<int, int, double>{ 16, 65, -4.60000002 },
       std::tuple<int, int, double>{ 16, 68, -4.60000002 },
       std::tuple<int, int, double>{ 16, 69, -4.60000002 },
       std::tuple<int, int, double>{ 16, 71, -4.60000002 },
       std::tuple<int, int, double>{ 16, 72, -4.60000002 },
       std::tuple<int, int, double>{ 16, 74, -4.60000002 },
       std::tuple<int, int, double>{ 16, 109, -1.0 },
       std::tuple<int, int, double>{ 16, 110, 1.0 },
       std::tuple<int, int, double>{ 16, 129, -1.0 },
       std::tuple<int, int, double>{ 16, 130, 1.0 },
       std::tuple<int, int, double>{ 16, 149, 1.0 },
       std::tuple<int, int, double>{ 16, 150, -1.0 },
       std::tuple<int, int, double>{ 16, 169, 1.0 },
       std::tuple<int, int, double>{ 16, 170, -1.0 },
       std::tuple<int, int, double>{ 17, 53, -4.60000002 },
       std::tuple<int, int, double>{ 17, 57, -4.60000002 },
       std::tuple<int, int, double>{ 17, 60, -4.60000002 },
       std::tuple<int, int, double>{ 17, 62, -4.60000002 },
       std::tuple<int, int, double>{ 17, 64, -4.60000002 },
       std::tuple<int, int, double>{ 17, 66, -4.60000002 },
       std::tuple<int, int, double>{ 17, 68, -4.60000002 },
       std::tuple<int, int, double>{ 17, 70, -4.60000002 },
       std::tuple<int, int, double>{ 17, 71, -4.60000002 },
       std::tuple<int, int, double>{ 17, 73, -4.60000002 },
       std::tuple<int, int, double>{ 17, 74, -4.60000002 },
       std::tuple<int, int, double>{ 17, 113, -1.0 },
       std::tuple<int, int, double>{ 17, 114, 1.0 },
       std::tuple<int, int, double>{ 17, 133, -1.0 },
       std::tuple<int, int, double>{ 17, 134, 1.0 },
       std::tuple<int, int, double>{ 17, 153, 1.0 },
       std::tuple<int, int, double>{ 17, 154, -1.0 },
       std::tuple<int, int, double>{ 17, 173, 1.0 },
       std::tuple<int, int, double>{ 17, 174, -1.0 },
       std::tuple<int, int, double>{ 18, 54, -4.60000002 },
       std::tuple<int, int, double>{ 18, 58, -4.60000002 },
       std::tuple<int, int, double>{ 18, 61, -4.60000002 },
       std::tuple<int, int, double>{ 18, 63, -4.60000002 },
       std::tuple<int, int, double>{ 18, 64, -4.60000002 },
       std::tuple<int, int, double>{ 18, 67, -4.60000002 },
       std::tuple<int, int, double>{ 18, 69, -4.60000002 },
       std::tuple<int, int, double>{ 18, 70, -4.60000002 },
       std::tuple<int, int, double>{ 18, 72, -4.60000002 },
       std::tuple<int, int, double>{ 18, 73, -4.60000002 },
       std::tuple<int, int, double>{ 18, 74, -4.60000002 },
       std::tuple<int, int, double>{ 18, 117, -1.0 },
       std::tuple<int, int, double>{ 18, 118, 1.0 },
       std::tuple<int, int, double>{ 18, 137, -1.0 },
       std::tuple<int, int, double>{ 18, 138, 1.0 },
       std::tuple<int, int, double>{ 18, 157, 1.0 },
       std::tuple<int, int, double>{ 18, 158, -1.0 },
       std::tuple<int, int, double>{ 18, 177, 1.0 },
       std::tuple<int, int, double>{ 18, 178, -1.0 },
       std::tuple<int, int, double>{ 19, 75, -4.60000002 },
       std::tuple<int, int, double>{ 19, 80, -4.60000002 },
       std::tuple<int, int, double>{ 19, 81, -4.60000002 },
       std::tuple<int, int, double>{ 19, 82, -4.60000002 },
       std::tuple<int, int, double>{ 19, 83, -4.60000002 },
       std::tuple<int, int, double>{ 19, 90, -4.60000002 },
       std::tuple<int, int, double>{ 19, 91, -4.60000002 },
       std::tuple<int, int, double>{ 19, 92, -4.60000002 },
       std::tuple<int, int, double>{ 19, 93, -4.60000002 },
       std::tuple<int, int, double>{ 19, 94, -4.60000002 },
       std::tuple<int, int, double>{ 19, 95, -4.60000002 },
       std::tuple<int, int, double>{ 19, 102, -1.0 },
       std::tuple<int, int, double>{ 19, 103, 1.0 },
       std::tuple<int, int, double>{ 19, 122, -1.0 },
       std::tuple<int, int, double>{ 19, 123, 1.0 },
       std::tuple<int, int, double>{ 19, 142, 1.0 },
       std::tuple<int, int, double>{ 19, 143, -1.0 },
       std::tuple<int, int, double>{ 19, 162, 1.0 },
       std::tuple<int, int, double>{ 19, 163, -1.0 },
       std::tuple<int, int, double>{ 20, 76, -4.60000002 },
       std::tuple<int, int, double>{ 20, 80, -4.60000002 },
       std::tuple<int, int, double>{ 20, 84, -4.60000002 },
       std::tuple<int, int, double>{ 20, 85, -4.60000002 },
       std::tuple<int, int, double>{ 20, 86, -4.60000002 },
       std::tuple<int, int, double>{ 20, 90, -4.60000002 },
       std::tuple<int, int, double>{ 20, 91, -4.60000002 },
       std::tuple<int, int, double>{ 20, 92, -4.60000002 },
       std::tuple<int, int, double>{ 20, 96, -4.60000002 },
       std::tuple<int, int, double>{ 20, 97, -4.60000002 },
       std::tuple<int, int, double>{ 20, 98, -4.60000002 },
       std::tuple<int, int, double>{ 20, 106, -1.0 },
       std::tuple<int, int, double>{ 20, 107, 1.0 },
       std::tuple<int, int, double>{ 20, 126, -1.0 },
       std::tuple<int, int, double>{ 20, 127, 1.0 },
       std::tuple<int, int, double>{ 20, 146, 1.0 },
       std::tuple<int, int, double>{ 20, 147, -1.0 },
       std::tuple<int, int, double>{ 20, 166, 1.0 },
       std::tuple<int, int, double>{ 20, 167, -1.0 },
       std::tuple<int, int, double>{ 21, 77, -4.60000002 },
       std::tuple<int, int, double>{ 21, 81, -4.60000002 },
       std::tuple<int, int, double>{ 21, 84, -4.60000002 },
       std::tuple<int, int, double>{ 21, 87, -4.60000002 },
       std::tuple<int, int, double>{ 21, 88, -4.60000002 },
       std::tuple<int, int, double>{ 21, 90, -4.60000002 },
       std::tuple<int, int, double>{ 21, 93, -4.60000002 },
       std::tuple<int, int, double>{ 21, 94, -4.60000002 },
       std::tuple<int, int, double>{ 21, 96, -4.60000002 },
       std::tuple<int, int, double>{ 21, 97, -4.60000002 },
       std::tuple<int, int, double>{ 21, 99, -4.60000002 },
       std::tuple<int, int, double>{ 21, 110, -1.0 },
       std::tuple<int, int, double>{ 21, 111, 1.0 },
       std::tuple<int, int, double>{ 21, 130, -1.0 },
       std::tuple<int, int, double>{ 21, 131, 1.0 },
       std::tuple<int, int, double>{ 21, 150, 1.0 },
       std::tuple<int, int, double>{ 21, 151, -1.0 },
       std::tuple<int, int, double>{ 21, 170, 1.0 },
       std::tuple<int, int, double>{ 21, 171, -1.0 },
       std::tuple<int, int, double>{ 22, 78, -4.60000002 },
       std::tuple<int, int, double>{ 22, 82, -4.60000002 },
       std::tuple<int, int, double>{ 22, 85, -4.60000002 },
       std::tuple<int, int, double>{ 22, 87, -4.60000002 },
       std::tuple<int, int, double>{ 22, 89, -4.60000002 },
       std::tuple<int, int, double>{ 22, 91, -4.60000002 },
       std::tuple<int, int, double>{ 22, 93, -4.60000002 },
       std::tuple<int, int, double>{ 22, 95, -4.60000002 },
       std::tuple<int, int, double>{ 22, 96, -4.60000002 },
       std::tuple<int, int, double>{ 22, 98, -4.60000002 },
       std::tuple<int, int, double>{ 22, 99, -4.60000002 },
       std::tuple<int, int, double>{ 22, 114, -1.0 },
       std::tuple<int, int, double>{ 22, 115, 1.0 },
       std::tuple<int, int, double>{ 22, 134, -1.0 },
       std::tuple<int, int, double>{ 22, 135, 1.0 },
       std::tuple<int, int, double>{ 22, 154, 1.0 },
       std::tuple<int, int, double>{ 22, 155, -1.0 },
       std::tuple<int, int, double>{ 22, 174, 1.0 },
       std::tuple<int, int, double>{ 22, 175, -1.0 },
       std::tuple<int, int, double>{ 23, 79, -4.60000002 },
       std::tuple<int, int, double>{ 23, 83, -4.60000002 },
       std::tuple<int, int, double>{ 23, 86, -4.60000002 },
       std::tuple<int, int, double>{ 23, 88, -4.60000002 },
       std::tuple<int, int, double>{ 23, 89, -4.60000002 },
       std::tuple<int, int, double>{ 23, 92, -4.60000002 },
       std::tuple<int, int, double>{ 23, 94, -4.60000002 },
       std::tuple<int, int, double>{ 23, 95, -4.60000002 },
       std::tuple<int, int, double>{ 23, 97, -4.60000002 },
       std::tuple<int, int, double>{ 23, 98, -4.60000002 },
       std::tuple<int, int, double>{ 23, 99, -4.60000002 },
       std::tuple<int, int, double>{ 23, 118, -1.0 },
       std::tuple<int, int, double>{ 23, 119, 1.0 },
       std::tuple<int, int, double>{ 23, 138, -1.0 },
       std::tuple<int, int, double>{ 23, 139, 1.0 },
       std::tuple<int, int, double>{ 23, 158, 1.0 },
       std::tuple<int, int, double>{ 23, 159, -1.0 },
       std::tuple<int, int, double>{ 23, 178, 1.0 },
       std::tuple<int, int, double>{ 23, 179, -1.0 },
   };
   Vec<std::string> rnames{
       "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11", "12", "13",
       "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
   };
   Vec<std::string> cnames{
       "A1",   "B1",   "C1",   "D1",   "E1",   "AB1",  "AC1",  "AD1",  "AE1",
       "BC1",  "BD1",  "BE1",  "CD1",  "CE1",  "DE1",  "ABC1", "ABD1", "ABE1",
       "ACD1", "ACE1", "ADE1", "BCD1", "BCE1", "BDE1", "CDE1", "A2",   "B2",
       "C2",   "D2",   "E2",   "AB2",  "AC2",  "AD2",  "AE2",  "BC2",  "BD2",
       "BE2",  "CD2",  "CE2",  "DE2",  "ABC2", "ABD2", "ABE2", "ACD2", "ACE2",
       "ADE2", "BCD2", "BCE2", "BDE2", "CDE2", "A3",   "B3",   "C3",   "D3",
       "E3",   "AB3",  "AC3",  "AD3",  "AE3",  "BC3",  "BD3",  "BE3",  "CD3",
       "CE3",  "DE3",  "ABC3", "ABD3", "ABE3", "ACD3", "ACE3", "ADE3", "BCD3",
       "BCE3", "BDE3", "CDE3", "A4",   "B4",   "C4",   "D4",   "E4",   "AB4",
       "AC4",  "AD4",  "AE4",  "BC4",  "BD4",  "BE4",  "CD4",  "CE4",  "DE4",
       "ABC4", "ABD4", "ABE4", "ACD4", "ACE4", "ADE4", "BCD4", "BCE4", "BDE4",
       "CDE4", "TA1",  "TA2",  "TA3",  "TA4",  "TB1",  "TB2",  "TB3",  "TB4",
       "TC1",  "TC2",  "TC3",  "TC4",  "TD1",  "TD2",  "TD3",  "TD4",  "TE1",
       "TE2",  "TE3",  "TE4",  "UA1",  "UA2",  "UA3",  "UA4",  "UB1",  "UB2",
       "UB3",  "UB4",  "UC1",  "UC2",  "UC3",  "UC4",  "UD1",  "UD2",  "UD3",
       "UD4",  "UE1",  "UE2",  "UE3",  "UE4",  "VA1",  "VA2",  "VA3",  "VA4",
       "VB1",  "VB2",  "VB3",  "VB4",  "VC1",  "VC2",  "VC3",  "VC4",  "VD1",
       "VD2",  "VD3",  "VD4",  "VE1",  "VE2",  "VE3",  "VE4",  "WA1",  "WA2",
       "WA3",  "WA4",  "WB1",  "WB2",  "WB3",  "WB4",  "WC1",  "WC2",  "WC3",
       "WC4",  "WD1",  "WD2",  "WD3",  "WD4",  "WE1",  "WE2",  "WE3",  "WE4",
   };
   int64_t nCols = 180;
   int64_t nRows = 24;
   ProblemBuilder<double> pb;
   pb.reserve( 460, 24, 180 );
   pb.setNumRows( nRows );
   pb.setNumCols( nCols );
   pb.setObjAll( coeffobj );
   pb.setObjOffset( 0.0 );
   pb.setColLbAll( lbs );
   pb.setColLbInfAll( lbInf );
   pb.setColUbAll( ubs );
   pb.setColUbInfAll( ubInf );
   pb.setColIntegralAll( isIntegral );
   pb.setRowLhsInfAll( lhsIsInf );
   pb.setRowRhsInfAll( rhsIsInf );
   pb.setRowLhsAll( lhs );
   pb.setRowRhsAll( rhs );
   pb.setRowNameAll( rnames );
   pb.addEntryAll( entries );
   pb.setColNameAll( cnames );
   pb.setProblemName( "rgn.hpp" );
   Problem<double> problem = pb.build();
   /// PROBLEM BUILDER CODE END

   return problem;
}

} // namespace instances
} // namespace papilo

#endif
