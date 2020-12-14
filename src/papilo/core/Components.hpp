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

#ifndef _PAPILO_CORE_COMPONENTS_HPP_
#define _PAPILO_CORE_COMPONENTS_HPP_

#include "papilo/core/Problem.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Vec.hpp"
#include "pdqsort/pdqsort.h"
#include <boost/pending/disjoint_sets.hpp>

namespace papilo
{

struct ComponentInfo
{
   int64_t componentid;
   int64_t nintegral;
   int64_t ncontinuous;
   int64_t nnonz;

   bool
   operator<( const ComponentInfo& other ) const
   {
      return std::make_tuple( nintegral, nnonz, ncontinuous, componentid ) <
             std::make_tuple( other.nintegral, other.nnonz, other.ncontinuous,
                              other.componentid );
   }
};

class Components
{
 private:
   Vec<int64_t> col2comp;
   Vec<int64_t> row2comp;
   Vec<int64_t> compcols;
   Vec<int64_t> comprows;
   Vec<int64_t> compcolstart;
   Vec<int64_t> comprowstart;
   Vec<ComponentInfo> compInfo;

 public:
   const int64_t*
   getComponentsRows( int64_t c ) const
   {
      return &comprows[comprowstart[c]];
   }

   int64_t
   getComponentsNumRows( int64_t c ) const
   {
      return comprowstart[c + 1] - comprowstart[c];
   }

   int64_t
   getRowComponentIdx( int64_t row ) const
   {
      return row2comp[row];
   }

   const int64_t*
   getComponentsCols( int64_t c ) const
   {
      return &compcols[compcolstart[c]];
   }

   int64_t
   getComponentsNumCols( int64_t c ) const
   {
      return compcolstart[c + 1] - compcolstart[c];
   }

   int64_t
   getColComponentIdx( int64_t col ) const
   {
      return col2comp[col];
   }

   const Vec<ComponentInfo>&
   getComponentInfo() const
   {
      return compInfo;
   }

   template <typename REAL>
   int64_t
   detectComponents( const Problem<REAL>& problem )
   {
      const int64_t ncols = problem.getNCols();
      std::unique_ptr<int64_t[]> rank{ new int64_t[ncols] };
      std::unique_ptr<int64_t[]> parent{ new int64_t[ncols] };
      boost::disjoint_sets<int64_t*, int64_t*> djsets( rank.get(), parent.get() );

      for( int64_t i = 0; i != ncols; ++i )
         djsets.make_set( i );

      const ConstraintMatrix<REAL>& consMatrix = problem.getConstraintMatrix();
      const IndexRange* ranges;
      int64_t nrows;

      std::tie( ranges, nrows ) = consMatrix.getRangeInfo();
      const int64_t* colinds = consMatrix.getColumns();

      for( int64_t r = 0; r != nrows; ++r )
      {
         if( ranges[r].end - ranges[r].start <= 1 )
            continue;

         int64_t firstcol = colinds[ranges[r].start];

         for( int64_t i = ranges[r].start + 1; i != ranges[r].end; ++i )
            djsets.link( firstcol, colinds[i] );
      }

      HashMap<int64_t, int64_t> componentmap;

      for( int64_t i = 0; i != ncols; ++i )
      {
         int64_t nextid = static_cast<int64_t>( componentmap.size() );
         auto insert_result =
             componentmap.insert( { djsets.find_set( i ), nextid } );
      }

      int64_t numcomponents = static_cast<int64_t>( componentmap.size() );

      if( numcomponents > 1 )
      {
         col2comp.resize( ncols );
         compcols.resize( ncols );

         for( int64_t i = 0; i != ncols; ++i )
         {
            col2comp[i] = componentmap[djsets.find_set( i )];
            compcols[i] = i;
         }

         row2comp.resize( nrows );
         comprows.resize( nrows );
         for( int64_t i = 0; i != nrows; ++i )
         {
            assert( problem.getConstraintMatrix()
                        .getRowCoefficients( i )
                        .getLength() > 0 );
            int64_t col = problem.getConstraintMatrix()
                          .getRowCoefficients( i )
                          .getIndices()[0];
            row2comp[i] = col2comp[col];
            comprows[i] = i;
         }

         pdqsort( compcols.begin(), compcols.end(), [&]( int64_t col1, int64_t col2 ) {
            return col2comp[col1] < col2comp[col2];
         } );

         compcolstart.resize( numcomponents + 1 );
         int64_t k = 0;

         // first component starts at 0
         compcolstart[0] = 0;

         // fill out starts for second to last components, also reuse the
         // col2comp vector to map the columns of a component to indices
         // starting at 0 without gaps
         for( int64_t i = 1; i != numcomponents; ++i )
         {
            while( k != ncols && col2comp[compcols[k]] == i - 1 )
            {
               col2comp[compcols[k]] = k - compcolstart[i - 1];
               ++k;
            }

            compcolstart[i] = k;
         }

         while( k != ncols )
         {
            assert( col2comp[compcols[k]] == numcomponents - 1 );
            col2comp[compcols[k]] = k - compcolstart[numcomponents - 1];
            ++k;
         }

         // last component ends at ncols
         compcolstart[numcomponents] = ncols;

         pdqsort( comprows.begin(), comprows.end(), [&]( int64_t row1, int64_t row2 ) {
            return row2comp[row1] < row2comp[row2];
         } );

         comprowstart.resize( numcomponents + 1 );
         k = 0;
         // first component starts at 0
         comprowstart[0] = 0;

         // fill out starts for second to last components, also reuse the
         // row2comp vector to map the rows of a component to indices starting
         // at 0 without gaps
         for( int64_t i = 1; i != numcomponents; ++i )
         {
            while( k != nrows && row2comp[comprows[k]] == i - 1 )
            {
               row2comp[comprows[k]] = k - comprowstart[i - 1];
               ++k;
            }

            comprowstart[i] = k;
         }

         while( k != nrows )
         {
            assert( row2comp[comprows[k]] == numcomponents - 1 );
            row2comp[comprows[k]] = k - comprowstart[numcomponents - 1];
            ++k;
         }
         // last component ends at nrows
         comprowstart[numcomponents] = nrows;

         // compute size informaton of components
         compInfo.resize( numcomponents );
         const auto& colsizes = problem.getColSizes();
         const auto& cflags = problem.getColFlags();

         for( int64_t i = 0; i != numcomponents; ++i )
         {
            for( int64_t j = compcolstart[i]; j != compcolstart[i + 1]; ++j )
            {
               if( cflags[compcols[j]].test( ColFlag::kIntegral ) )
                  ++compInfo[i].nintegral;
               else
                  ++compInfo[i].ncontinuous;

               compInfo[i].nnonz += colsizes[compcols[j]];
               compInfo[i].componentid = i;
            }
         }

         pdqsort( compInfo.begin(), compInfo.end() );
      }

      return numcomponents;
   }
};

} // namespace papilo

#endif