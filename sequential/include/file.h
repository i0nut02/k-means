#ifndef FILE_H
#define FILE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

#include "const.h"


/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename);

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples);

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data);

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename);

/* 
function writeLog: It writes on the filename file the variable message
*/
int writeLog(const char* filename, const char* message);

#ifdef __cplusplus
}
#endif

#endif
