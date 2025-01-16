/* TA-LIB Copyright (c) 1999-2024, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *  PP       Peter Pudaite
 *  AA       Andrew Atkinson
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  120802 MF   Template creation.
 *  101103 PP   Initial creation of code.
 *  112603 MF   Add independent control to the RSI period.
 *  020605 AA   Fix #1117656. NULL pointer assignement.
 */

/**** START GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
/* All code within this section is automatically
 * generated by gen_code. Any modification will be lost
 * next time gen_code is run.
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */    #include "TA-Lib-Core.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode::InternalError)
/* Generated */    namespace TicTacTec { namespace TA { namespace Library {
/* Generated */ #elif defined( _JAVA )
/* Generated */    #include "ta_defs.h"
/* Generated */    #include "ta_java_defs.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode.InternalError)
/* Generated */ #elif defined( _RUST )
/* Generated */    #include "ta_defs.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode.InternalError)
/* Generated */ #else
/* Generated */    #include <string.h>
/* Generated */    #include <math.h>
/* Generated */    #include "ta_func.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_UTILITY_H
/* Generated */    #include "ta_utility.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_MEMORY_H
/* Generated */    #include "ta_memory.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #define TA_PREFIX(x) TA_##x
/* Generated */ #define INPUT_TYPE   double
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ int Core::StochRsiLookback( int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                           int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                           int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                           MAType        optInFastD_MAType ) /* Generated */ 
/* Generated */ #elif defined( _JAVA )
/* Generated */ public int stochRsiLookback( int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                            int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                            int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                            MAType        optInFastD_MAType ) /* Generated */ 
/* Generated */ #else
/* Generated */ TA_LIB_API int TA_STOCHRSI_Lookback( int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                               int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                               int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                               TA_MAType     optInFastD_MAType ) /* Generated */ 
/* Generated */ #endif
/**** END GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
{
   /* insert local variable here */
   int retValue;

/**** START GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */    /* min/max are checked for optInTimePeriod. */
/* Generated */    if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInTimePeriod = 14;
/* Generated */    else if( ((int)optInTimePeriod < 2) || ((int)optInTimePeriod > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    /* min/max are checked for optInFastK_Period. */
/* Generated */    if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
/* Generated */       optInFastK_Period = 5;
/* Generated */    else if( ((int)optInFastK_Period < 1) || ((int)optInFastK_Period > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    /* min/max are checked for optInFastD_Period. */
/* Generated */    if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
/* Generated */       optInFastD_Period = 3;
/* Generated */    else if( ((int)optInFastD_Period < 1) || ((int)optInFastD_Period > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT )
/* Generated */       optInFastD_MAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInFastD_MAType < 0) || ((int)optInFastD_MAType > 8) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/**** END GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/

   /* insert lookback code here. */
   retValue = LOOKBACK_CALL(RSI)( optInTimePeriod ) + LOOKBACK_CALL(STOCHF)( optInFastK_Period, optInFastD_Period, optInFastD_MAType );

   return retValue;
}

/**** START GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
/*
 * TA_STOCHRSI - Stochastic Relative Strength Index
 * 
 * Input  = double
 * Output = double, double
 * 
 * Optional Parameters
 * -------------------
 * optInTimePeriod:(From 2 to 100000)
 *    Number of period
 * 
 * optInFastK_Period:(From 1 to 100000)
 *    Time period for building the Fast-K line
 * 
 * optInFastD_Period:(From 1 to 100000)
 *    Smoothing for making the Fast-D line. Usually set to 3
 * 
 * optInFastD_MAType:
 *    Type of Moving Average for Fast-D
 * 
 * 
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED ) && defined( USE_SUBARRAY )
/* Generated */ enum class Core::RetCode Core::StochRsi( int    startIdx,
/* Generated */                                          int    endIdx,
/* Generated */                                          SubArray<double>^ inReal,
/* Generated */                                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                          MAType        optInFastD_MAType,
/* Generated */                                          [Out]int%    outBegIdx,
/* Generated */                                          [Out]int%    outNBElement,
/* Generated */                                          SubArray<double>^  outFastK,
/* Generated */                                          SubArray<double>^  outFastD )
/* Generated */ #elif defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::StochRsi( int    startIdx,
/* Generated */                                          int    endIdx,
/* Generated */                                          cli::array<double>^ inReal,
/* Generated */                                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                          MAType        optInFastD_MAType,
/* Generated */                                          [Out]int%    outBegIdx,
/* Generated */                                          [Out]int%    outNBElement,
/* Generated */                                          cli::array<double>^  outFastK,
/* Generated */                                          cli::array<double>^  outFastD )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode stochRsi( int    startIdx,
/* Generated */                          int    endIdx,
/* Generated */                          double       inReal[],
/* Generated */                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                          MAType        optInFastD_MAType,
/* Generated */                          MInteger     outBegIdx,
/* Generated */                          MInteger     outNBElement,
/* Generated */                          double        outFastK[],
/* Generated */                          double        outFastD[] )
/* Generated */ #else
/* Generated */ TA_LIB_API TA_RetCode TA_STOCHRSI( int    startIdx,
/* Generated */                                    int    endIdx,
/* Generated */                                               const double inReal[],
/* Generated */                                               int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                               int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                               int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                               TA_MAType     optInFastD_MAType,
/* Generated */                                               int          *outBegIdx,
/* Generated */                                               int          *outNBElement,
/* Generated */                                               double        outFastK[],
/* Generated */                                               double        outFastD[] )
/* Generated */ #endif
/**** END GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
{
   /* insert local variable here */
   ARRAY_REF(tempRSIBuffer);

   ENUM_DECLARATION(RetCode) retCode;
   int lookbackTotal, lookbackSTOCHF, tempArraySize;
   VALUE_HANDLE_INT(outBegIdx1);
   VALUE_HANDLE_INT(outBegIdx2);
   VALUE_HANDLE_INT(outNbElement1);

/**** START GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */ 
/* Generated */    /* Validate the requested output range. */
/* Generated */    if( startIdx < 0 )
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */    if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */ 
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */    #endif /* !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInTimePeriod. */
/* Generated */    if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInTimePeriod = 14;
/* Generated */    else if( ((int)optInTimePeriod < 2) || ((int)optInTimePeriod > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    /* min/max are checked for optInFastK_Period. */
/* Generated */    if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
/* Generated */       optInFastK_Period = 5;
/* Generated */    else if( ((int)optInFastK_Period < 1) || ((int)optInFastK_Period > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    /* min/max are checked for optInFastD_Period. */
/* Generated */    if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
/* Generated */       optInFastD_Period = 3;
/* Generated */    else if( ((int)optInFastD_Period < 1) || ((int)optInFastD_Period > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT )
/* Generated */       optInFastD_MAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInFastD_MAType < 0) || ((int)optInFastD_MAType > 8) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !outFastK )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    if( !outFastD )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_JAVA) */
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/* Generated */ 
/**** END GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/

   /* Insert TA function code here. */

   /* Stochastic RSI
    *
    * Reference: "Stochastic RSI and Dynamic Momentum Index"
    *            by Tushar Chande and Stanley Kroll
    *            Stock&Commodities V.11:5 (189-199)
    *
    * The TA-Lib version offer flexibility beyond what is explain
    * in the Stock&Commodities article.
    *
    * To calculate the "Unsmoothed stochastic RSI" with symetry like
    * explain in the article, keep the optInTimePeriod and optInFastK_Period
    * equal. Example:
    *
    *    unsmoothed stoch RSI 14 : optInTimePeriod   = 14
    *                              optInFastK_Period = 14
    *                              optInFastD_Period = 'x'
    *
    * The outFastK is the unsmoothed RSI discuss in the article.
    *
    * You can set the optInFastD_Period to smooth the RSI. The smooth
    * version will be found in outFastD. The outFastK will still contain
    * the unsmoothed stoch RSI. If you do not care about the smoothing of
    * the StochRSI, just leave optInFastD_Period to 1 and ignore outFastD.
    */

   VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
   VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);

   /* Adjust startIdx to account for the lookback period. */
   lookbackSTOCHF  = LOOKBACK_CALL(STOCHF)( optInFastK_Period, optInFastD_Period, optInFastD_MAType );
   lookbackTotal   = LOOKBACK_CALL(RSI)( optInTimePeriod ) + lookbackSTOCHF;

   if( startIdx < lookbackTotal )
      startIdx = lookbackTotal;

   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
   }

   VALUE_HANDLE_DEREF(outBegIdx) = startIdx;

   tempArraySize = (endIdx - startIdx) + 1 + lookbackSTOCHF;

   ARRAY_ALLOC( tempRSIBuffer, tempArraySize );

   retCode = FUNCTION_CALL(RSI)(startIdx-lookbackSTOCHF,
                                endIdx,
                                inReal,
                                optInTimePeriod,
                                VALUE_HANDLE_OUT(outBegIdx1),
                                VALUE_HANDLE_OUT(outNbElement1),
                                tempRSIBuffer);

   if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) || VALUE_HANDLE_GET(outNbElement1) == 0 )
   {
      ARRAY_FREE( tempRSIBuffer );
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return retCode;
   }

   retCode = FUNCTION_CALL_DOUBLE(STOCHF)(0,
                                          tempArraySize-1,
                                          tempRSIBuffer,
                                          tempRSIBuffer,
                                          tempRSIBuffer,
                                          optInFastK_Period,
                                          optInFastD_Period,
                                          optInFastD_MAType,
                                          VALUE_HANDLE_OUT(outBegIdx2),
                                          outNBElement,
                                          outFastK,
                                          outFastD);

   ARRAY_FREE( tempRSIBuffer );

   if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) || ((int)VALUE_HANDLE_DEREF(outNBElement)) == 0 )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return retCode;
   }

   return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
}

/**** START GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #define  USE_SINGLE_PRECISION_INPUT
/* Generated */ #if !defined( _MANAGED ) && !defined( _JAVA )
/* Generated */    #undef   TA_PREFIX
/* Generated */    #define  TA_PREFIX(x) TA_S_##x
/* Generated */ #endif
/* Generated */ #undef   INPUT_TYPE
/* Generated */ #define  INPUT_TYPE float
/* Generated */ #if defined( _MANAGED ) && defined( USE_SUBARRAY )
/* Generated */ enum class Core::RetCode Core::StochRsi( int    startIdx,
/* Generated */                                          int    endIdx,
/* Generated */                                          SubArray<float>^ inReal,
/* Generated */                                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                          MAType        optInFastD_MAType,
/* Generated */                                          [Out]int%    outBegIdx,
/* Generated */                                          [Out]int%    outNBElement,
/* Generated */                                          SubArray<double>^  outFastK,
/* Generated */                                          SubArray<double>^  outFastD )
/* Generated */ #elif defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::StochRsi( int    startIdx,
/* Generated */                                          int    endIdx,
/* Generated */                                          cli::array<float>^ inReal,
/* Generated */                                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                                          MAType        optInFastD_MAType,
/* Generated */                                          [Out]int%    outBegIdx,
/* Generated */                                          [Out]int%    outNBElement,
/* Generated */                                          cli::array<double>^  outFastK,
/* Generated */                                          cli::array<double>^  outFastD )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode stochRsi( int    startIdx,
/* Generated */                          int    endIdx,
/* Generated */                          float        inReal[],
/* Generated */                          int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                          int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                          int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                          MAType        optInFastD_MAType,
/* Generated */                          MInteger     outBegIdx,
/* Generated */                          MInteger     outNBElement,
/* Generated */                          double        outFastK[],
/* Generated */                          double        outFastD[] )
/* Generated */ #else
/* Generated */ TA_RetCode TA_S_STOCHRSI( int    startIdx,
/* Generated */                           int    endIdx,
/* Generated */                           const float  inReal[],
/* Generated */                           int           optInTimePeriod, /* From 2 to 100000 */
/* Generated */                           int           optInFastK_Period, /* From 1 to 100000 */
/* Generated */                           int           optInFastD_Period, /* From 1 to 100000 */
/* Generated */                           TA_MAType     optInFastD_MAType,
/* Generated */                           int          *outBegIdx,
/* Generated */                           int          *outNBElement,
/* Generated */                           double        outFastK[],
/* Generated */                           double        outFastD[] )
/* Generated */ #endif
/* Generated */ {
/* Generated */    ARRAY_REF(tempRSIBuffer);
/* Generated */    ENUM_DECLARATION(RetCode) retCode;
/* Generated */    int lookbackTotal, lookbackSTOCHF, tempArraySize;
/* Generated */    VALUE_HANDLE_INT(outBegIdx1);
/* Generated */    VALUE_HANDLE_INT(outBegIdx2);
/* Generated */    VALUE_HANDLE_INT(outNbElement1);
/* Generated */  #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */     if( startIdx < 0 )
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */     if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
/* Generated */        optInTimePeriod = 14;
/* Generated */     else if( ((int)optInTimePeriod < 2) || ((int)optInTimePeriod > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
/* Generated */        optInFastK_Period = 5;
/* Generated */     else if( ((int)optInFastK_Period < 1) || ((int)optInFastK_Period > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
/* Generated */        optInFastD_Period = 3;
/* Generated */     else if( ((int)optInFastD_Period < 1) || ((int)optInFastD_Period > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */     if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT )
/* Generated */        optInFastD_MAType = (TA_MAType)0;
/* Generated */     else if( ((int)optInFastD_MAType < 0) || ((int)optInFastD_MAType > 8) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !outFastK )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( !outFastD )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */  #endif 
/* Generated */    VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */    VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */    lookbackSTOCHF  = LOOKBACK_CALL(STOCHF)( optInFastK_Period, optInFastD_Period, optInFastD_MAType );
/* Generated */    lookbackTotal   = LOOKBACK_CALL(RSI)( optInTimePeriod ) + lookbackSTOCHF;
/* Generated */    if( startIdx < lookbackTotal )
/* Generated */       startIdx = lookbackTotal;
/* Generated */    if( startIdx > endIdx )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */    }
/* Generated */    VALUE_HANDLE_DEREF(outBegIdx) = startIdx;
/* Generated */    tempArraySize = (endIdx - startIdx) + 1 + lookbackSTOCHF;
/* Generated */    ARRAY_ALLOC( tempRSIBuffer, tempArraySize );
/* Generated */    retCode = FUNCTION_CALL(RSI)(startIdx-lookbackSTOCHF,
/* Generated */                                 endIdx,
/* Generated */                                 inReal,
/* Generated */                                 optInTimePeriod,
/* Generated */                                 VALUE_HANDLE_OUT(outBegIdx1),
/* Generated */                                 VALUE_HANDLE_OUT(outNbElement1),
/* Generated */                                 tempRSIBuffer);
/* Generated */    if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) || VALUE_HANDLE_GET(outNbElement1) == 0 )
/* Generated */    {
/* Generated */       ARRAY_FREE( tempRSIBuffer );
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return retCode;
/* Generated */    }
/* Generated */    retCode = FUNCTION_CALL_DOUBLE(STOCHF)(0,
/* Generated */                                           tempArraySize-1,
/* Generated */                                           tempRSIBuffer,
/* Generated */                                           tempRSIBuffer,
/* Generated */                                           tempRSIBuffer,
/* Generated */                                           optInFastK_Period,
/* Generated */                                           optInFastD_Period,
/* Generated */                                           optInFastD_MAType,
/* Generated */                                           VALUE_HANDLE_OUT(outBegIdx2),
/* Generated */                                           outNBElement,
/* Generated */                                           outFastK,
/* Generated */                                           outFastD);
/* Generated */    ARRAY_FREE( tempRSIBuffer );
/* Generated */    if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) || ((int)VALUE_HANDLE_DEREF(outNBElement)) == 0 )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return retCode;
/* Generated */    }
/* Generated */    return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */ }
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ }}} // Close namespace TicTacTec.TA.Lib
/* Generated */ #endif
/**** END GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/

