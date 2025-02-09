#include "../include/file.h"

void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case TOO_MUCH_COLUMNS_ERR:
            fprintf(stderr,"\tFile %s has too many columns.\n", filename);
            fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
            break;
        case READ_ERR:
            fprintf(stderr,"Error reading file: %s.\n", filename);
            break;
        case WRITE_ERR:
            fprintf(stderr,"Error writing file: %s.\n", filename);
            break;
    }
    fflush(stderr);    
}

int readInput(char* filename, int *lines, int *samples) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL) {
                contsamples++;
                ptr = strtok(NULL, delim);
            }        
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else {
        return READ_ERR;
    }
}

int readInput2(char* filename, float* data) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {         
            ptr = strtok(line, delim);
            while(ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    }
    else {
        return READ_ERR;
    }
}

int writeResult(int *classMap, int lines, const char* filename) {    
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL) {
        for(int i=0; i<lines; i++) {
            fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else {
        return WRITE_ERR;
    }
}

int writeLog(const char* filename, const char* message) {    
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL) {
        fprintf(fp, "%s", message);
        fclose(fp);   
        return 0;
    }
    else {
        return WRITE_ERR;
    }
}