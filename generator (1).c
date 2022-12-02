#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>

#define BUF_SIZE 80

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Введите все параметры при запуске программы [размер] [имя файла]");
        exit(0);
    }

    FILE *f;
    f = fopen(argv[2],"w+b");
    if(!f) {
        printf("ERROR: Error open or create file \n");
        exit(0);
    }
    printf("INFO: File %s successfully opened\n", argv[2]);

    int resInRows = atoi(argv[1]);

    for (int i = 0; i < resInRows;i++) {
        for (int j = 0; j < resInRows - 1; j++) {
            fprintf(f, "%d ", rand()%100);
        }
        // array
        fprintf(f, "%d", rand()%100);
        fprintf(f, "\n");
    }

    fprintf(f, "\n");
    for (int j = 0; j < resInRows - 1; j++) {
        fprintf(f, "%d ", rand()%100);
    }
    fprintf(f, "%d", rand()%100);
    fprintf(f, "\n");
    printf("Data successfully generated\n");
    fclose(f);
}