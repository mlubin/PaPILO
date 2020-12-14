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

#ifndef _PAPILO_CORE_REDUCTIONS_HPP_
#define _PAPILO_CORE_REDUCTIONS_HPP_

#include "papilo/misc/Vec.hpp"
#include <cassert>

namespace papilo
{

struct ColReduction
{
   enum
   {
      NONE = -1,
      OBJECTIVE = -2,
      LOWER_BOUND = -3,
      UPPER_BOUND = -4,
      FIXED = -5,
      LOCKED = -6,
      LOCKED_STRONG = -7,
      SUBSTITUTE = -8,
      BOUNDS_LOCKED = -9,
      REPLACE = -10,
      SUBSTITUTE_OBJ = -11,
      PARALLEL = -12,
      IMPL_INT = -13,
   };
};

struct RowReduction
{
   enum
   {
      NONE = -1,
      RHS = -2,
      LHS = -3,
      REDUNDANT = -4,
      LOCKED = -5,
      LOCKED_STRONG = -6,
      RHS_INF = -7,
      LHS_INF = -8,
      SPARSIFY = -9,
   };
};

template <typename REAL>
struct Reduction
{
   /// value stored in reduction. Meaning depends on the operation
   REAL newval;

   /// index of row or negative for column specific operations
   int64_t row;

   /// index of column or negative for row specific operations
   int64_t col;

   Reduction( REAL newval, int64_t row, int64_t col )
       : newval( newval ), row( row ), col( col )
   {
   }
};

template <typename REAL>
class Reductions
{
 public:
   void
   startTransaction()
   {
      assert( transactions.empty() || transactions.back().end >= 0 );

      const int64_t start = static_cast<int>( reductions.size() );
      transactions.emplace_back( start, -1 );
   }

   void
   endTransaction()
   {
      assert( !transactions.empty() && transactions.back().end == -1 );

      const int64_t end = static_cast<int>( reductions.size() );
      assert( end != transactions.back().start );
      transactions.back().end = end;
   }

   void
   changeMatrixEntry( int64_t row, int64_t col, REAL newval )
   {
      assert( row >= 0 && col >= 0 );
      reductions.emplace_back( newval, row, col );
   }

   void
   changeRowLHS( int64_t row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::LHS );
   }

   void
   changeRowRHS( int64_t row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::RHS );
   }

   void
   changeRowLHSInf( int64_t row )
   {
      reductions.emplace_back( 0.0, row, RowReduction::LHS_INF );
   }

   void
   changeRowRHSInf( int64_t row )
   {
      reductions.emplace_back( 0.0, row, RowReduction::RHS_INF );
   }

   void
   markRowRedundant( int64_t row )
   {
      reductions.emplace_back( REAL{ 0.0 }, row, RowReduction::REDUNDANT );
   }

   /// lock row, i.e. modifications that come before this transaction are
   /// conflicting but not modifications that come after this transaction
   void
   lockRow( int64_t row )
   {
      // locks are only valid inside a transaction
      assert( !transactions.empty() && transactions.back().end == -1 );
      // locks must come first within a transaction
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, row, RowReduction::LOCKED );
      ++transactions.back().nlocks;
   }

   /// lock row with a strong lock, i.e. modifications that come before or after
   /// this transaction are conflicting
   void
   lockRowStrong( int64_t row )
   {
      // locks are only valid inside a transaction
      assert( !transactions.empty() && transactions.back().end == -1 );
      // locks must come first within a transaction
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, row, RowReduction::LOCKED_STRONG );
      ++transactions.back().nlocks;
   }

   void
   changeObjCoeff( int64_t col, REAL newval )
   {
      reductions.emplace_back( newval, ColReduction::OBJECTIVE, col );
   }

   void
   changeColLB( int64_t col, REAL newval )
   {
      reductions.emplace_back( newval, ColReduction::LOWER_BOUND, col );
   }

   void
   changeColUB( int64_t col, REAL newval )
   {
      reductions.emplace_back( newval, ColReduction::UPPER_BOUND, col );
   }

   void
   fixCol( int64_t col, REAL val )
   {
      reductions.emplace_back( val, ColReduction::FIXED, col );
   }

   /// lock column, i.e. modifications that come before this transaction are
   /// conflicting but not modifications that come after this transaction
   void
   lockCol( int64_t col )
   {
      assert( !transactions.empty() && transactions.back().end == -1 );
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, ColReduction::LOCKED, col );
      ++transactions.back().nlocks;
   }

   /// lock column with a strong lock, i.e. modifications that come before or
   /// after this transaction are conflicting
   void
   lockColStrong( int64_t col )
   {
      assert( !transactions.empty() && transactions.back().end == -1 );
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, ColReduction::LOCKED_STRONG, col );
      ++transactions.back().nlocks;
   }

   /// lock column lower and upper bounds
   void
   lockColBounds( int64_t col )
   {
      assert( !transactions.empty() && transactions.back().end == -1 );
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, ColReduction::BOUNDS_LOCKED, col );
      ++transactions.back().nlocks;
   }

   /// signal that a column in free and can be substituted in the matrix
   void
   aggregateFreeCol( int64_t col, int64_t equalityRow )
   {
      assert( col >= 0 && equalityRow >= 0 );
      reductions.emplace_back( static_cast<REAL>( equalityRow ),
                               ColReduction::SUBSTITUTE, col );
   }

   /// signal that a column in free and can be substituted in the matrix
   void
   substituteColInObjective( int64_t col, int64_t equalityRow )
   {
      assert( col >= 0 && equalityRow >= 0 );
      reductions.emplace_back( static_cast<REAL>( equalityRow ),
                               ColReduction::SUBSTITUTE_OBJ, col );
   }

   // replace col1 = factor * col2 + offset
   void
   replaceCol( int64_t col1, int64_t col2, REAL factor, REAL offset )
   {
      assert( col1 >= 0 && col2 >= 0 );

      startTransaction();
      reductions.emplace_back( factor, ColReduction::REPLACE, col1 );
      reductions.emplace_back( offset, ColReduction::NONE, col2 );
      endTransaction();
   }

   /// parallel columns col1 and col2 must satisfies all conditions so
   /// that they can be substituted by a new variable y = col2 + factor * col1
   /// where factor is computed by using the ratio between the two
   /// columns coefficients
   void
   parallelCols( int64_t col1, int64_t col2 )
   {
      assert( col1 >= 0 && col2 >= 0 );
      reductions.emplace_back( static_cast<REAL>( col2 ),
                               ColReduction::PARALLEL, col1 );
   }

   void
   impliedInteger( int64_t col )
   {
      assert( col >= 0 );
      reductions.emplace_back( 0, ColReduction::IMPL_INT, col );
   }

   void
   sparsify( int64_t eq, int64_t numrows, const std::pair<int, REAL>* sparsifiedrows )
   {
      reductions.emplace_back( static_cast<REAL>( numrows ), eq,
                               RowReduction::SPARSIFY );
      for( int64_t i = 0; i != numrows; ++i )
         reductions.emplace_back( sparsifiedrows[i].second,
                                  sparsifiedrows[i].first, RowReduction::NONE );
   }

   unsigned int
   size()
   {
      return reductions.size();
   }

   void
   clear()
   {
      reductions.clear();
      transactions.clear();
   }

   const Vec<Reduction<REAL>>&
   getReductions() const
   {
      return reductions;
   }

   struct Transaction
   {
      int64_t start;
      int64_t end;
      int64_t nlocks;
      int64_t naddcoeffs;

      Transaction( int64_t start, int64_t end )
          : start( start ), end( end ), nlocks( 0 ), naddcoeffs( 0 )
      {
      }
   };

   const Vec<Transaction>&
   getTransactions() const
   {
      return transactions;
   }

 private:
   Vec<Reduction<REAL>> reductions;
   Vec<Transaction> transactions;

 public:
   Reduction<REAL>&
   getReduction( int64_t i )
   {
      return reductions[i];
   }
};

template <typename REAL>
class TransactionGuard
{
 public:
   TransactionGuard( Reductions<REAL>& reductions ) : reductions( reductions )
   {
      reductions.startTransaction();
   }

   ~TransactionGuard() { reductions.endTransaction(); }

 private:
   Reductions<REAL>& reductions;
};

} // namespace papilo

#endif
