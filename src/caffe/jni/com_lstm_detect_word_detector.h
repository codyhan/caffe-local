/* DO NOT EDIT THIS FILE - it is machine generated */
#include "jni.h"
/* Header for class com_lstm_detect_word_detector */

#ifndef _Included_com_lstm_detect_word_detector
#define _Included_com_lstm_detect_word_detector
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_lstm_detect_word_detector
 * Method:    detect_box
 * Signature: (Ljava/lang/String;FF)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_lstm_detect_word_1detector_detect_1box
  (JNIEnv *, jclass, jstring, jfloat, jfloat);

/*
 * Class:     com_lstm_detect_word_detector
 * Method:    detect_word
 * Signature: (JIII)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_lstm_detect_word_1detector_detect_1word
  (JNIEnv *, jclass, jlong, jint, jint, jint);

#ifdef __cplusplus
}
#endif
#endif
