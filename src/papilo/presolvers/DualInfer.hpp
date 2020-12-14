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

#ifndef _PAPILO_PRESOLVERS_DUAL_INFER_HPP_
#define _PAPILO_PRESOLVERS_DUAL_INFER_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/core/SingleRow.hpp"

namespace papilo
{

template <typename REAL>
class DualInfer : public PresolveMethod<REAL>
{
 public:
   DualInfer() : PresolveMethod<REAL>()
   {
      this->setName( "dualinfer" );
      this->setTiming( PresolverTiming::kExhaustive );
      this->setType( PresolverType::kContinuousCols );
   }

   bool
   initialize( const Problem<REAL>& problem,
               const PresolveOptions& presolveOptions ) override
   {
      if( presolveOptions.dualreds == 0 )
         this->setEnabled( false );
      return false;
   }

   /// todo how to communicate about postsolve information
   virtual PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions ) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class DualInfer<double>;
extern template class DualInfer<Quad>;
extern template class DualInfer<Rational>;
#endif

template <typename REAL>
PresolveStatus
DualInfer<REAL>::execute( const Problem<REAL>& problem,
                          const ProblemUpdate<REAL>& problemUpdate,
                          const Num<REAL>& num, Reductions<REAL>& reductions )
{
   assert( problem.getNumContinuousCols() != 0 );

   const auto& obj = problem.getObjective().coefficients;
   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& lbValues = problem.getLowerBounds();
   const auto& ubValues = problem.getUpperBounds();
   const auto& lhsValues = consMatrix.getLeftHandSides();
   const auto& rhsValues = consMatrix.getRightHandSides();
   const auto& rflags = consMatrix.getRowFlags();
   const auto& cflags = problem.getColFlags();
   const auto& rowsize = consMatrix.getRowSizes();
   const auto& colsize = consMatrix.getColSizes();
   const int64_t ncols = problem.getNCols();
   const int64_t nrows = problem.getNRows();
   const auto& activities = problem.getRowActivities();

   PresolveStatus result = PresolveStatus::kUnchanged;

   // get called less and less over time regardless of success since the
   // presolver can be expensive otherwise
   this->skipRounds( this->getNCalls() );

   Vec<RowActivity<REAL>> activitiesCopy( activities );
   auto checkNonImpliedBounds = [&]( int64_t col, bool& lbinf, bool& ubinf ) {
      const REAL& lb = lbValues[col];
      const REAL& ub = ubValues[col];
      ColFlags colf = cflags[col];

      auto colvec = consMatrix.getColumnCoefficients( col );
      const int64_t len = colvec.getLength();
      const int64_t* inds = colvec.getIndices();
      const REAL* vals = colvec.getValues();

      int64_t i = 0;
      while(
          ( !colf.test( ColFlag::kLbInf ) || !colf.test( ColFlag::kUbInf ) ) &&
          i != len )
      {
         int64_t row = inds[i];

         assert( !consMatrix.isRowRedundant( row ) );

         bool lhsinf = rflags[row].test( RowFlag::kLhsInf );
         bool rhsinf = rflags[row].test( RowFlag::kRhsInf );
         const REAL& lhs = lhsValues[row];
         const REAL& rhs = rhsValues[row];

         if( !colf.test( ColFlag::kUbInf ) )
         {
            if( row_implies_UB( num, lhs, rhs, rflags[row], activitiesCopy[row],
                                vals[i], lb, ub, colf ) )
            {
               colf.set( ColFlag::kUbInf );

               if( !colf.test( ColFlag::kUbHuge ) )
                  update_activities_remove_finite_bound( inds, vals, len,
                                                         BoundChange::kUpper,
                                                         ub, activitiesCopy );
            }
         }

         if( !colf.test( ColFlag::kLbInf ) )
         {
            if( row_implies_LB( num, lhs, rhs, rflags[row], activitiesCopy[row],
                                vals[i], lb, ub, colf ) )
            {
               colf.set( ColFlag::kLbInf );

               if( !colf.test( ColFlag::kLbHuge ) )
                  update_activities_remove_finite_bound( inds, vals, len,
                                                         BoundChange::kLower,
                                                         lb, activitiesCopy );
            }
         }

         ++i;
      }

      lbinf = colf.test( ColFlag::kLbInf );
      ubinf = colf.test( ColFlag::kUbInf );
   };

   // initialize dual variable domains
   Vec<REAL> dualLB;
   Vec<REAL> dualUB;
   Vec<ColFlags> dualColFlags( nrows );

   dualLB.resize( nrows, REAL{ 0 } );
   dualUB.resize( nrows, REAL{ 0 } );

   for( int64_t i = 0; i != nrows; ++i )
   {
      if( consMatrix.isRowRedundant( i ) )
         continue;

      assert( rowsize[i] > 0 );

      if( !rflags[i].test( RowFlag::kRhsInf ) )
         dualColFlags[i].set( ColFlag::kLbInf );
      if( !rflags[i].test( RowFlag::kLhsInf ) )
         dualColFlags[i].set( ColFlag::kUbInf );
   }

   // initialize dual constraint sides, mark constraints of integral columns
   // redundant to skip them for propagation
   const Vec<REAL>& dualLHS = obj;
   const Vec<REAL>& dualRHS = obj;
   Vec<RowFlags> dualRowFlags( ncols );

   Vec<std::pair<int, int>> checkRedundantBounds;
   checkRedundantBounds.reserve( ncols );
   const Vec<int64_t>& colperm = problemUpdate.getRandomColPerm();
   for( int64_t c = 0; c != ncols; ++c )
   {
      assert( !dualRowFlags[c].test( RowFlag::kLhsInf ) );
      assert( !dualRowFlags[c].test( RowFlag::kRhsInf ) );

      if( cflags[c].test( ColFlag::kIntegral, ColFlag::kInactive ) ||
          colsize[c] == 0 )
      {
         dualRowFlags[c].set( RowFlag::kRhsInf );
         dualRowFlags[c].set( RowFlag::kLhsInf );
         dualRowFlags[c].set( RowFlag::kRedundant );
         continue;
      }

      int64_t colweight = int64_t( !cflags[c].test( ColFlag::kLbInf ) +
                                   !cflags[c].test( ColFlag::kUbInf ) ) *
                              colsize[c] * ncols +
                          colperm[c];

      checkRedundantBounds.emplace_back(
          int( std::min( int64_t( std::numeric_limits<int>::max() ),
                         colweight ) ),
          c );
   }

   pdqsort( checkRedundantBounds.begin(), checkRedundantBounds.end(),
            []( std::pair<int, int> c1, std::pair<int, int> c2 ) {
               return c1.first < c2.first;
            } );

   for( std::pair<int, int> pair : checkRedundantBounds )
   {
      int64_t c = pair.second;
      assert( colsize[c] > 0 );

      bool lbinf;
      bool ubinf;
      checkNonImpliedBounds( c, lbinf, ubinf );

      if( !lbinf && !ubinf )
      {
         dualRowFlags[c].set( RowFlag::kRhsInf );
         dualRowFlags[c].set( RowFlag::kLhsInf );
         dualRowFlags[c].set( RowFlag::kRedundant );
         continue;
      }

      if( !lbinf )
         dualRowFlags[c].set( RowFlag::kLhsInf );

      if( !ubinf )
         dualRowFlags[c].set( RowFlag::kRhsInf );
   }

   // compute initial activities, skip redundant rows
   Vec<RowActivity<REAL>> dualActivities( ncols );

   Vec<int64_t> changedActivity;
   Vec<int64_t> nextChangedActivity;

   auto checkRedundancy = [&]( int64_t dualRow ) {
      if( ( dualRowFlags[dualRow].test( RowFlag::kLhsInf ) ||
            ( dualActivities[dualRow].ninfmin == 0 &&
              num.isFeasGE( dualActivities[dualRow].min,
                            dualLHS[dualRow] ) ) ) &&
          ( dualRowFlags[dualRow].test( RowFlag::kRhsInf ) ||
            ( dualActivities[dualRow].ninfmax == 0 &&
              num.isFeasLE( dualActivities[dualRow].max,
                            dualRHS[dualRow] ) ) ) )
         return true;
      return false;
   };

   for( int64_t i = 0; i != ncols; ++i )
   {
      if( dualRowFlags[i].test( RowFlag::kRedundant ) )
      {
         dualActivities[i].min = 0;
         dualActivities[i].max = 0;
         dualActivities[i].ninfmin = colsize[i];
         dualActivities[i].ninfmax = colsize[i];
         continue;
      }

      auto colvec = consMatrix.getColumnCoefficients( i );

      dualActivities[i] = compute_row_activity(
          colvec.getValues(), colvec.getIndices(), colvec.getLength(), dualLB,
          dualUB, dualColFlags );

      if( checkRedundancy( i ) )
      {
         dualRowFlags[i].set( RowFlag::kRedundant );
         continue;
      }

      if( ( dualActivities[i].ninfmax <= 1 &&
            !dualRowFlags[i].test( RowFlag::kLhsInf ) ) ||
          ( dualActivities[i].ninfmin <= 1 &&
            !dualRowFlags[i].test( RowFlag::kRhsInf ) ) )
         changedActivity.push_back( i );
   }

   // do domain propagation
   int64_t nrounds = 0;
   auto boundChanged = [&]( BoundChange boundChg, int64_t dualCol, REAL newbound ) {
      auto rowvec = consMatrix.getRowCoefficients( dualCol );

      bool oldboundinf;
      REAL oldbound;

      if( boundChg == BoundChange::kLower )
      {
         oldboundinf = dualColFlags[dualCol].test( ColFlag::kLbInf );
         oldbound = dualLB[dualCol];

         // check against other bound for infeasibility
         if( !dualColFlags[dualCol].test( ColFlag::kUbInf ) )
         {
            REAL bnddist = dualUB[dualCol] - newbound;

            // bound exceeded by more then feastol means infeasible
            if( bnddist < -num.getFeasTol() )
            {
               result = PresolveStatus::kUnbndOrInfeas;
               return;
            }

            // bound is equal, or almost equal
            if( bnddist <= num.getFeasTol() )
            {
               bool forcingrow = true;
               if( bnddist > 0 )
               {
                  // bound is almost equal, check if fixing the column is
                  // numerically negligible
                  auto dualColVec = consMatrix.getRowCoefficients( dualCol );
                  REAL maxabsval = dualColVec.getMaxAbsValue();
                  if( bnddist * maxabsval > num.getFeasTol() )
                     forcingrow = false;
               }

               // fix column
               if( forcingrow )
                  newbound = dualUB[dualCol];
            }
         }

         // reject too small bound changes
         if( !oldboundinf && newbound - oldbound <= +1000 * num.getFeasTol() )
            return;

         dualColFlags[dualCol].unset( ColFlag::kLbInf );
         dualLB[dualCol] = newbound;

         assert( dualColFlags[dualCol].test( ColFlag::kUbInf ) ||
                 dualLB[dualCol] <= dualUB[dualCol] );
      }
      else
      {
         oldboundinf = dualColFlags[dualCol].test( ColFlag::kUbInf );
         oldbound = dualUB[dualCol];

         // check against other bound for infeasibility
         if( !dualColFlags[dualCol].test( ColFlag::kLbInf ) )
         {
            REAL bnddist = newbound - dualLB[dualCol];

            // bound exceeded by more then feastol means infeasible
            if( bnddist < -num.getFeasTol() )
            {
               result = PresolveStatus::kUnbndOrInfeas;
               return;
            }

            if( bnddist <= num.getFeasTol() )
            {
               bool forcingrow = true;
               if( bnddist > 0 )
               {
                  auto dualColVec = consMatrix.getRowCoefficients( dualCol );
                  REAL maxabsval = dualColVec.getMaxAbsValue();
                  if( bnddist * maxabsval > num.getFeasTol() )
                     forcingrow = false;
               }

               if( forcingrow )
                  newbound = dualLB[dualCol];
            }
         }

         // reject too small bound changes
         if( !oldboundinf && newbound - oldbound >= -1000 * num.getFeasTol() )
            return;

         dualColFlags[dualCol].unset( ColFlag::kUbInf );
         dualUB[dualCol] = newbound;

         assert( dualColFlags[dualCol].test( ColFlag::kLbInf ) ||
                 dualUB[dualCol] >= dualLB[dualCol] );
      }

      const REAL* vals = rowvec.getValues();
      const int64_t* inds = rowvec.getIndices();
      const int64_t len = rowvec.getLength();

      for( int64_t i = 0; i != len; ++i )
      {
         int64_t dualRow = inds[i];

         if( dualRowFlags[dualRow].test( RowFlag::kRedundant ) )
            continue;

         RowActivity<REAL>& activity = dualActivities[dualRow];

         ActivityChange actChange = update_activity_after_boundchange(
             vals[i], boundChg, oldbound, newbound, oldboundinf, activity );

         if( checkRedundancy( dualRow ) )
         {
            dualRowFlags[dualRow].set( RowFlag::kRedundant );
            continue;
         }

         if( activity.lastchange != nrounds )
         {
            if( actChange == ActivityChange::kMin &&
                !dualRowFlags[dualRow].test( RowFlag::kRhsInf ) &&
                activity.ninfmin <= 1 )
            {
               activity.lastchange = nrounds;
               nextChangedActivity.push_back( dualRow );
            }
            else if( actChange == ActivityChange::kMax &&
                     !dualRowFlags[dualRow].test( RowFlag::kLhsInf ) &&
                     activity.ninfmax <= 1 )
            {
               activity.lastchange = nrounds;
               nextChangedActivity.push_back( dualRow );
            }
         }
      }
   };

   using std::swap;
   while( !changedActivity.empty() )
   {
      Message::debug( this, "dual progation round {} on {} dual rows\n",
                      nrounds, changedActivity.size() );
      for( int64_t dualRow : changedActivity )
      {
         if( dualRowFlags[dualRow].test( RowFlag::kRedundant ) )
            continue;

         auto colvec = consMatrix.getColumnCoefficients( dualRow );

         propagate_row( colvec.getValues(), colvec.getIndices(),
                        colvec.getLength(), dualActivities[dualRow],
                        dualLHS[dualRow], dualRHS[dualRow],
                        dualRowFlags[dualRow], dualLB, dualUB, dualColFlags,
                        boundChanged );

         if( result == PresolveStatus::kUnbndOrInfeas )
            return result;
      }

      swap( changedActivity, nextChangedActivity );
      nextChangedActivity.clear();
      ++nrounds;
   }

   // analyze dual domains:
   int64_t impliedeqs = 0;
   int64_t fixedints = 0;
   int64_t fixedconts = 0;

   // change inequality rows to equations if their dual value cannot be zero
   for( int64_t i = 0; i < nrows; ++i )
   {
      if( consMatrix.isRowRedundant( i ) )
         continue;

      if( !rflags[i].test( RowFlag::kLhsInf ) &&
          !rflags[i].test( RowFlag::kRhsInf ) && lhsValues[i] == rhsValues[i] )
         continue;

      if( ( !dualColFlags[i].test( ColFlag::kLbInf ) &&
            num.isFeasGT( dualLB[i], 0 ) ) )
      {
         assert( !rflags[i].test( RowFlag::kLhsInf ) );

         if( activities[i].ninfmax != 0 ||
             num.isFeasLT( lhsValues[i], activities[i].max ) )
         {
            TransactionGuard<REAL> tg{ reductions };
            reductions.lockRow( i );
            reductions.changeRowRHS( i, lhsValues[i] );
            ++impliedeqs;
         }
      }
      else if( !dualColFlags[i].test( ColFlag::kUbInf ) &&
               num.isFeasLT( dualUB[i], 0 ) )
      {
         assert( !rflags[i].test( RowFlag::kRhsInf ) );

         if( activities[i].ninfmin != 0 ||
             num.isFeasGT( rhsValues[i], activities[i].min ) )
         {
            TransactionGuard<REAL> tg{ reductions };
            reductions.lockRow( i );
            reductions.changeRowLHS( i, rhsValues[i] );
            ++impliedeqs;
         }
      }
   }

   // use dualActivities to compute reduced cost bounds and use them to fix
   // variables
   for( int64_t i = 0; i < ncols; ++i )
   {
      if( colsize[i] <= 0 )
         continue;

      if( dualRowFlags[i].test( RowFlag::kRedundant ) )
      {
         auto colvec = consMatrix.getColumnCoefficients( i );

         dualActivities[i] = compute_row_activity(
             colvec.getValues(), colvec.getIndices(), colvec.getLength(),
             dualLB, dualUB, dualColFlags );
      }

      if( dualActivities[i].ninfmax == 0 &&
          num.isFeasLT( dualActivities[i].max, obj[i] ) &&
          num.isSafeLT( dualActivities[i].max, obj[i] ) )
      {
         assert( !cflags[i].test( ColFlag::kLbInf ) );
         TransactionGuard<REAL> tg{ reductions };
         reductions.lockColBounds( i );
         reductions.fixCol( i, lbValues[i] );

         if( cflags[i].test( ColFlag::kIntegral ) )
            ++fixedints;
         else
            ++fixedconts;
      }
      else if( dualActivities[i].ninfmin == 0 &&
               num.isFeasGT( dualActivities[i].min, obj[i] ) &&
               num.isSafeGT( dualActivities[i].min, obj[i] ) )
      {
         assert( !cflags[i].test( ColFlag::kUbInf ) );
         TransactionGuard<REAL> tg{ reductions };
         reductions.lockColBounds( i );
         reductions.fixCol( i, ubValues[i] );

         if( cflags[i].test( ColFlag::kIntegral ) )
            ++fixedints;
         else
            ++fixedconts;
      }
   }

   // set result if reductions where found
   if( impliedeqs > 0 || fixedints > 0 || fixedconts > 0 )
      result = PresolveStatus::kReduced;

   // if( nrounds != 0 && result == PresolveStatus::kReduced )
   //    fmt::print(
   //        "   Dual infer: {} propagation rounds, {} implied equations, "
   //        "{} integers fixed, {} conts fixed\n",
   //        nrounds, impliedeqs, fixedints, fixedconts );

   return result;
}

} // namespace papilo

#endif
