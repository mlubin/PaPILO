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

#ifndef PAPILO_TEST_INSTANCES_FLUGPL
#define PAPILO_TEST_INSTANCES_FLUGPL

#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

namespace papilo
{
namespace instances
{

Problem<double>
flugpl()
{
   /// PROBLEM BUILDER CODE
   Vec<double> coeffobj{
       2700.0, 1500.0, 30.0, 2700.0, 1500.0, 30.0, 2700.0, 1500.0, 30.0,
       2700.0, 1500.0, 30.0, 2700.0, 1500.0, 30.0, 2700.0, 1500.0, 30.0,
   };
   Vec<double> lbs{
       0.0,  0.0, 0.0, 57.0, 0.0, 0.0, 57.0, 0.0, 0.0,
       57.0, 0.0, 0.0, 57.0, 0.0, 0.0, 57.0, 0.0, 0.0,
   };
   Vec<uint8_t> lbInf{
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   };
   Vec<double> ubs{
       0.0,  18.0, 0.0, 75.0, 18.0, 0.0, 75.0, 18.0, 0.0,
       75.0, 18.0, 0.0, 75.0, 18.0, 0.0, 75.0, 18.0, 0.0,
   };
   Vec<uint8_t> ubInf{
       1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
   };
   Vec<uint8_t> isIntegral{
       0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
   };
   Vec<uint8_t> lhsIsInf{
       0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
   };
   Vec<double> lhs{
       60.0, 8000.0,  0.0, 0.0, 9000.0, 0.0, 0.0, 8000.0,  0.0,
       0.0,  10000.0, 0.0, 0.0, 9000.0, 0.0, 0.0, 12000.0, 0.0,
   };
   Vec<uint8_t> rhsIsInf{
       0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
   };
   Vec<double> rhs{
       60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 150.0 },
       std::tuple<int, int, double>{ 1, 1, -100.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 2, 0, -20.0 },
       std::tuple<int, int, double>{ 2, 2, 1.0 },
       std::tuple<int, int, double>{ 3, 0, 0.9 },
       std::tuple<int, int, double>{ 3, 1, 1.0 },
       std::tuple<int, int, double>{ 3, 3, -1.0 },
       std::tuple<int, int, double>{ 4, 3, 150.0 },
       std::tuple<int, int, double>{ 4, 4, -100.0 },
       std::tuple<int, int, double>{ 4, 5, 1.0 },
       std::tuple<int, int, double>{ 5, 3, -20.0 },
       std::tuple<int, int, double>{ 5, 5, 1.0 },
       std::tuple<int, int, double>{ 6, 3, 0.9 },
       std::tuple<int, int, double>{ 6, 4, 1.0 },
       std::tuple<int, int, double>{ 6, 6, -1.0 },
       std::tuple<int, int, double>{ 7, 6, 150.0 },
       std::tuple<int, int, double>{ 7, 7, -100.0 },
       std::tuple<int, int, double>{ 7, 8, 1.0 },
       std::tuple<int, int, double>{ 8, 6, -20.0 },
       std::tuple<int, int, double>{ 8, 8, 1.0 },
       std::tuple<int, int, double>{ 9, 6, 0.9 },
       std::tuple<int, int, double>{ 9, 7, 1.0 },
       std::tuple<int, int, double>{ 9, 9, -1.0 },
       std::tuple<int, int, double>{ 10, 9, 150.0 },
       std::tuple<int, int, double>{ 10, 10, -100.0 },
       std::tuple<int, int, double>{ 10, 11, 1.0 },
       std::tuple<int, int, double>{ 11, 9, -20.0 },
       std::tuple<int, int, double>{ 11, 11, 1.0 },
       std::tuple<int, int, double>{ 12, 9, 0.9 },
       std::tuple<int, int, double>{ 12, 10, 1.0 },
       std::tuple<int, int, double>{ 12, 12, -1.0 },
       std::tuple<int, int, double>{ 13, 12, 150.0 },
       std::tuple<int, int, double>{ 13, 13, -100.0 },
       std::tuple<int, int, double>{ 13, 14, 1.0 },
       std::tuple<int, int, double>{ 14, 12, -20.0 },
       std::tuple<int, int, double>{ 14, 14, 1.0 },
       std::tuple<int, int, double>{ 15, 12, 0.9 },
       std::tuple<int, int, double>{ 15, 13, 1.0 },
       std::tuple<int, int, double>{ 15, 15, -1.0 },
       std::tuple<int, int, double>{ 16, 15, 150.0 },
       std::tuple<int, int, double>{ 16, 16, -100.0 },
       std::tuple<int, int, double>{ 16, 17, 1.0 },
       std::tuple<int, int, double>{ 17, 15, -20.0 },
       std::tuple<int, int, double>{ 17, 17, 1.0 },
   };
   Vec<std::string> rnames{
       "ANZ1", "STD1", "UEB1", "ANZ2", "STD2", "UEB2", "ANZ3", "STD3", "UEB3",
       "ANZ4", "STD4", "UEB4", "ANZ5", "STD5", "UEB5", "ANZ6", "STD6", "UEB6",
   };
   Vec<std::string> cnames{
       "STM1", "ANM1", "UE1", "STM2", "ANM2", "UE2", "STM3", "ANM3", "UE3",
       "STM4", "ANM4", "UE4", "STM5", "ANM5", "UE5", "STM6", "ANM6", "UE6",
   };
   int64_t nCols = 18;
   int64_t nRows = 18;
   ProblemBuilder<double> pb;
   pb.reserve( 46, 18, 18 );
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
   pb.setProblemName( "flugpl.hpp" );
   Problem<double> problem = pb.build();
   /// PROBLEM BUILDER CODE END

   return problem;
}

} // namespace instances
} // namespace papilo

#endif
