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

#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/Objective.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/io/MpsParser.hpp"
#include "papilo/misc/fmt.hpp"

using namespace papilo;

// Returns True if cols in given permutation are same
static bool
check_cols( const VariableDomains<double> vd1, const VariableDomains<double> vd2, Vec<int> perm1, const Vec<int> perm2 )
{
   assert(perm1.size() == perm2.size() );
   int ncols = perm1.size();

   for( int i = 0; i < ncols; ++i )
   {
      int i1 = perm1[i];
      int i2 = perm2[i];

      if( vd1.flags[i1].test( ColFlag::kIntegral ) !=
          vd2.flags[i2].test( ColFlag::kIntegral ) )
      {
         // kein duplikat: eine variable ist ganzzahlig die andere nicht
         fmt::print("One variable is integer the other not in col prob1:{} and prob2:{}\n", i1, i2);
         return false;
      }

      if( vd1.flags[i1].test( ColFlag::kUbInf ) !=
          vd2.flags[i2].test( ColFlag::kUbInf ) )
      {
         // kein duplikat: ein upper bound ist +infinity, der andere nicht
         fmt::print("One variable's UB is +infty the others not in col prob1:{} and prob2:{}\n", i1, i2);
         return false;
      }

      if( vd1.flags[i1].test( ColFlag::kLbInf ) !=
          vd2.flags[i2].test( ColFlag::kLbInf ) )
      {
         // kein duplikat: ein lower bound ist -infinity, der andere nicht
         fmt::print("One variable's UB is -infty the other not in col prob1:{} and prob2:{}\n", i1, i2);
         return false;
      }

      if( !vd1.flags[i1].test( ColFlag::kLbInf ) &&
          vd1.lower_bounds[i1] != vd2.lower_bounds[i2] )
      {
         assert( !vd2.flags[i2].test( ColFlag::kLbInf ) );
         // kein duplikat: lower bounds sind endlich aber unterschiedlich
         fmt::print("LBs are different in col prob1:{} and prob2:{}\n", i1, i2);
         return false;
      }

      if( !vd1.flags[i1].test( ColFlag::kUbInf ) &&
          vd1.upper_bounds[i1] != vd2.upper_bounds[i2] )
      {
         assert( !vd2.flags[i2].test( ColFlag::kUbInf ) );
         // kein duplikat: upper bounds sind endlich aber unterschiedlich
         fmt::print("UBs are different in col prob1:{} and prob2:{}\n", i1, i2);
         return false;
      }
   }
   return true;
}

// Returns True if rows are in
static bool
check_rows( const ConstraintMatrix<double> cm1, const ConstraintMatrix<double> cm2, Vec<int> permrow1, Vec<int> permrow2, Vec<int> permcol1, Vec<int> permcol2 )
{
   assert(permrow1.size() == permrow2.size());
   assert(permcol1.size() == permcol2.size());
   int nrows = permrow1.size();
   int ncols = permcol1.size();

   for( int i = 0; i < nrows; ++i)
   {
      int i1row = permrow1[i];
      int i2row = permrow2[i];

      // Check Row coefficients
      const SparseVectorView<double> row1 = cm1.getRowCoefficients(i1row);
      const SparseVectorView<double> row2 = cm2.getRowCoefficients(i2row);

      // Assume: If there is different amounts of variables in constraint it is not the same (not entirely true)
      const int curr_ncols = row1.getLength();
      if( curr_ncols != row2.getLength() )
      {
         fmt::print("Different amounts of variables in row prob1:{} and prob2:{}\n", i1row, i2row);
         return false;
      }

      const int* inds1 = row1.getIndices();
      const int* inds2 = row2.getIndices();
      const double* vals1 = row1.getValues();
      const double* vals2 = row2.getValues();

      for( int x = 0; x < curr_ncols; ++x)
      {
         // Check if same variables are defined for row
         int final_index1 = permcol1[*(inds1 + x)];
         int final_index2 = permcol2[*(inds2 + x)];
         if( final_index1 != final_index2)
         {
            fmt::print("Different columns defined in row prob1:{} and prob2:{}\n", i1row, i2row);
            return false;
         }
         // Check if values are same
         if( *(vals1 + x) != *(vals2 + x) )
         {
            fmt::print("Different coefficients in row prob1:{} and prob2:{}\n", i1row, i2row);
            return false;
         }
      }
   }
   return true;
}

static bool
check_duplicates( const Problem<double>& prob1, const Problem<double>& prob2 )
{
   // Check for columns
   int ncols = prob1.getNCols();

   if( ncols != prob2.getNCols() )
   {
      // kein duplikat: unterschiedlich viele variablen
      fmt::print("not same number of variables\n");
      return false;
   }

   const VariableDomains<double>& vd1 = prob1.getVariableDomains();
   const VariableDomains<double>& vd2 = prob2.getVariableDomains();

   Vec<int> nocolperm(ncols);
   // std::iota(noperm.begin(), noperm.end(), 0);
   std::generate(nocolperm.begin(), nocolperm.end(), [] {
      static int i = 0;
      return i++;
   });

   if( !check_cols(vd1, vd2, nocolperm, nocolperm) ) return false;

   // Check for rows
   // First assume for being same you need to have same rows (even though not true)
   int nrows = prob1.getNRows();

   if( nrows != prob2.getNRows() )
   {
      fmt::print("not same number of rows: prob1:{} prob2:{}\n", nrows, prob2.getNRows() );
      return false;
   }

   Vec<int> norowperm(nrows);
   std::generate(norowperm.begin(), norowperm.end(), [] {
      static int i = 0;
      return i++;
   });

   const ConstraintMatrix<double> cm1 = prob1.getConstraintMatrix();
   const ConstraintMatrix<double> cm2 = prob2.getConstraintMatrix();

   if( !check_rows(cm1, cm2, norowperm, norowperm, nocolperm, nocolperm) ) return false;

   // All checks passed
   return true;
}

int
main( int argc, char* argv[] )
{

   assert( argc == 3 );

   Problem<double> prob1 = MpsParser<double>::loadProblem( argv[1] );
   Problem<double> prob2 = MpsParser<double>::loadProblem( argv[2] );

   bool res = check_duplicates( prob1, prob2 );
   fmt::print( "duplicates: {}\n", res);

   return 0;
}