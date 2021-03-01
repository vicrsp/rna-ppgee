/*
 * dpp_two_class.c
 * Copyright (C) Manuel Fernandez Delgado 2011 <manuel.fernandez.delgado@usc.es>
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANPOILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//////////////////
// dpp_two_class.c => Direct Calculation of Parallel Perceptron weights
// Author: Manuel Fernandez Delgado
//////////////////
// makefile:
// CC = gcc
// OPCS = -g
// 
// dpp_two_class : dpp_two_class.c
// 	$(CC) $(OPCS) dpp_two_class.c -o dpp_two_class -lm
////////////////

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <sys/timeb.h>
#include <unistd.h>


#define N_INPUTS_ORIGINAL 13
#define N_INPUTS 14  // debe ser N_INPUTS_ORIGINAL + 1
#define N_CLASES 2
#define DIR_DATA "."
#define DIR_RESULTS "."
#define N_PERCEPTRONS 3
#define N_TRIALS 10
#define N_INIC 50
#define PORC_TRAIN_PATTERNS 90
#define SRANDOM_KEY 1237907981L


FILE  *pf_results;
float  accuracy_trial[N_TRIALS];
float  accuracy_med;
float  accuracy_desv;
float  accuracy_entrenamento;
float  **w;
float  **wb;  // best weight
float  **x1;
float  **x;
int  y;  // real output for the current pattern
int  *d;  // desired outputs for Parallel Perceptron
int  **dik;	// desiref outputs for single perceptrons
float  media[N_INPUTS];
float  stdev[N_INPUTS];
int  trial;
int  n_patterns_total;
int  n_patterns_train;
int  n_patterns_valid;
int  pa;  // patron_actual
int  *patterns_train;
int  *patterns_valid;
int  inic;
char  f_datos[200];
char  f_resultados[200];
struct timeb  tempo_inicial;
struct timeb  tempo_final;


void start_time() {
	ftime(&tempo_inicial);
}


void evaluate_time() {

	ftime(&tempo_final);
	time_t  tempo_transcurrido = tempo_final.time - tempo_inicial.time;
	time_t  horas = tempo_transcurrido/3600;
	time_t  minutos = (tempo_transcurrido - horas*3600)/60;
	time_t  segundos = tempo_transcurrido - horas*3600 - minutos*60;
	short  mseg = tempo_final.millitm - tempo_inicial.millitm;
	if(mseg < 0) {
		mseg = 1000 + mseg;
		segundos--; tempo_transcurrido--;
	} else if(mseg >= 1000) {
		do {
			mseg = mseg - 1000;
			segundos++; tempo_transcurrido++;
		} while(mseg >= 1000);
	}
	printf("tempo transcurrido = %i.%i seg (%i h, %i min, %i seg, %i mseg.)\n", tempo_transcurrido, mseg, horas, minutos, segundos, mseg);
	fprintf(pf_results, "tempo transcurrido = %i.%i seg (%i h, %i min, %i seg, %i mseg.)\n", tempo_transcurrido, mseg, horas, minutos, segundos, mseg);

}



void le_n_patrons() {
	FILE  *pf;
	char  c;


	pf = fopen(f_datos, "r");
	if(! pf) {
		fprintf(stderr, "dpp.c: le_n_patrons: erro en fopen abrindo %s: %s\n", f_datos, strerror(errno));
		exit(1);
	}
	n_patterns_total = 0;
	while(! feof(pf)) {
		fscanf(pf, "%c", &c);
		if('\n' == c) {
			n_patterns_total++;
		}
	}
	n_patterns_total--;
	fclose(pf);
}



void alloc_memory() {
	int  i;


	w = (float **) calloc(N_PERCEPTRONS, sizeof(float *));
	wb = (float **) calloc(N_PERCEPTRONS, sizeof(float *));
	for(i = 0; i < N_PERCEPTRONS; i++) {
		w[i] = (float *) calloc(N_INPUTS, sizeof(float));
		wb[i] = (float *) calloc(N_INPUTS, sizeof(float));
	}

	le_n_patrons();
	x1 = (float **) calloc(n_patterns_total, sizeof(float *));
	x = (float **) calloc(n_patterns_total, sizeof(float *));
	for(i = 0; i < n_patterns_total; i++) {
		x1[i] = (float *) calloc(N_INPUTS, sizeof(float));
		x[i] = (float *) calloc(N_INPUTS, sizeof(float));
	}
	d = (int *) calloc(n_patterns_total, sizeof(int));
	dik = (int **) calloc(N_PERCEPTRONS, sizeof(int *));
	for(i = 0; i < N_PERCEPTRONS; i++) {
		dik[i] = (int *) calloc(n_patterns_total, sizeof(int));
	}

	n_patterns_train = (float) n_patterns_total*PORC_TRAIN_PATTERNS/100;
	n_patterns_valid = n_patterns_total - n_patterns_train;

	patterns_train = (int *) calloc(n_patterns_train, sizeof(int));
	patterns_valid = (int *) calloc(n_patterns_valid, sizeof(int));
}



void read_data() {
	FILE  *pf;
	int  i;
	int  j;


	pf = fopen(f_datos, "r");
	if(! pf) {
		fprintf(stderr, "dpp.c: read_data: erro en fopen abrindo %s: %s\n", f_datos, strerror(errno));
		exit(1);
	}
	for(i = 0; i < n_patterns_total; i++) {
		for(j = 0; j < N_INPUTS_ORIGINAL; j++) {
			fscanf(pf, "%f,", &x1[i][j]);
		}
		x1[i][N_INPUTS_ORIGINAL] = 1;
		fscanf(pf, "%i\n", &d[i]);  // d[i] must be -1 or +1
	}
	fclose(pf);
}


void inicializa() {
	srandom((unsigned int) SRANDOM_KEY);
	sprintf(f_datos, "%s/data.dat", DIR_DATA);
	sprintf(f_resultados, "%s/results_dpp.dat", DIR_RESULTS);
	alloc_memory();
	read_data();
	pf_results = fopen(f_resultados, "w");
	if(! pf_results) {
		fprintf(stderr, "erro abrindo results_dpp.dat: %s\n", strerror(errno));
		exit(1);
	}
	start_time();
}



// a media e stdev calculanse so sobre os patrons de entrenamento
void evaluate_data() {
	float  u;
	float  t;
	int  i;
	int  j;


	for(i = 0; i < N_INPUTS; i++) {
		for(j = 0, u = 0; j < n_patterns_train; j++) {
			pa = patterns_train[j];
			u += x1[pa][i];
		}
		media[i] = u/n_patterns_train;
		for(j = u = 0; j < n_patterns_train; j++) {
			pa = patterns_train[j];
			t = x1[pa][i] - media[i];
			u += t*t;
		}
		stdev[i] = sqrt(u/n_patterns_train);
	}
}



/**
\brief preprocessing de modo que para cada componhente do patron a media sexa 0 e a desviacion estandar 1 
*/
void preprocessing() {
	int  i;
	int  j;


	// preprocessing media 0 e desviacion 1
	for(i = 0; i < N_INPUTS; i++) {
		if(stdev[i]) {  
			for(j = 0; j < n_patterns_total; j++) {
				x[j][i] = (x1[j][i] - media[i])/stdev[i];
			}
		} else {  // a entrada N_INPUTS_ORIGINAL ten stdev == 0: non se preprocesa
			for(j = 0; j < n_patterns_total; j++) {
				x[j][i] = x1[j][i];
			}
		}
	}
}



void generate_patterns() {
	int  i;
	int  t;
	char  u[n_patterns_total];


	bzero(u, n_patterns_total*sizeof(char));
	for(i = 0; i < n_patterns_valid;) {
		t = (float) n_patterns_total*random()/RAND_MAX;
		if(! u[t]) {
			patterns_valid[i++] = t; u[t] = 1;
		}
	}
	for(i = 0; i < n_patterns_train;) {
		t = (float) n_patterns_total*random()/RAND_MAX;
		if(! u[t]) {
			patterns_train[i++] = t; u[t] = 1;
		}
	}
}


int perceptron(int n, int p) {
	float  y;
	int  i;


	for(i = y = 0; i < N_INPUTS; i++) {
		y += w[n][i]*x[p][i];
	}
	return(y >= 0 ? 1 : -1);
}


int output_pp(int p) {
	int  i;
	int  sum;


	for(i = sum = 0; i < N_PERCEPTRONS; i++) {
		sum += perceptron(i, p);
	}
	return(sum >= 0 ? 1 : -1);
}






// xeracion das d_ik[i][k]
void inicializa_dik() {
	int  i;
	int  j;
	int  k;
	int  n;
	int  y;


	n = N_PERCEPTRONS/2;
	for(k = 0; k < n_patterns_total; k++) {
		for(i = 0; i < N_PERCEPTRONS; i++) {
			dik[i][k] = d[k];
		}
		for(j = 0; j < n; j++) {  // 0 <= y <= N_PERCEPTRONS - 1
			y = N_PERCEPTRONS*(random() - 1.)/RAND_MAX;
			dik[y][k] = -dik[y][k];
		}
	}
}



void calculate_weights() {
	float  t;
	float  norma_z;
	float  z[N_INPUTS];
	int  i;
	int  j;
	int  k;


	inicializa_dik();

	for(i = 0; i < N_PERCEPTRONS; i++) {
		// calculo de vector z
		for(j = norma_z = 0; j < N_INPUTS; j++) {
			for(k = t = 0; k < n_patterns_train; k++) {
				pa = patterns_train[k];
				t += dik[i][pa]*x[pa][j];
			}
			z[j] = t; norma_z += t*t;
		}
		norma_z = sqrt(norma_z);

		// calculo de vector w
		for(j = 0; j < N_INPUTS; j++) {
			t = z[j]/norma_z;
			w[i][j] = t;
		}
	}

	// calculo do accuracy de clasificación sobre o conxunto de entrenamento
	for(k = t = 0; k < n_patterns_train; k++) {
		pa = patterns_train[k];
		y = output_pp(pa);
		if(y == d[pa]) {
			t++;
		}
	}
	t = 100.*t/n_patterns_train;
	printf("train accuracy= %.1f%%\n", t);

	if(t > accuracy_trial[trial]) {
		accuracy_trial[trial] = t;
		for(i = 0; i < N_PERCEPTRONS; i++) {
			memcpy(&wb[i], &w[i], N_INPUTS*sizeof(float));
		}
	}
}


void validate() {
	float  t;
	int  i;


	// copy the best weight vector wb to w
	for(i = 0; i < N_PERCEPTRONS; i++) {
		memcpy(&w[i], &wb[i], N_INPUTS*sizeof(float));
	}

	for(i = t = 0; i < n_patterns_valid; i++) {
		pa = patterns_valid[i];
		y = output_pp(pa);
		if(y == d[pa]) {
			t++;
		}
	}
	t = 100*t/n_patterns_valid;
	printf("trial %i/%i: inic %i/%i: validando ... accuracy= %.1f%%\n", trial, N_TRIALS, inic, N_INIC, t);
	fprintf(pf_results, "trial %i/%i: inic %i/%i: validando ... accuracy= %.1f%%\n", trial, N_TRIALS, inic, N_INIC, t);

}





void average_results() {
	float  t;
	int  i;


	printf("RESULTS:\n");
	fprintf(pf_results, "RESULTS:\n");
	for(i = accuracy_med = 0; i < N_TRIALS; i++) {
		accuracy_med += accuracy_trial[i];
		printf("trial=%i accuracy= %.1f%%\n", i, accuracy_trial[i]);
		fprintf(pf_results, "trial=%i accuracy= %.1f%%\n", i, accuracy_trial[i]);
	}
	accuracy_med /= N_TRIALS;
	for(i = accuracy_desv = 0; i < N_TRIALS; i++) {
		t = accuracy_trial[i] - accuracy_med;
		accuracy_desv += t*t;
	}
	accuracy_desv = sqrt(accuracy_desv/N_TRIALS);
	printf("accuracy_med= %.1f%%(%g)\n", accuracy_med, accuracy_desv);
	fprintf(pf_results, "accuracy_med= %.1f%%(%g)\n", accuracy_med, accuracy_desv);
}




void free_memory() {
	int  i;


	for(i = 0; i < N_PERCEPTRONS; i++) {
		free(w[i]);
		free(wb[i]);
	}
	free(w);
	free(wb);

	free(patterns_train);
	free(patterns_valid);
	for(i = 0; i < n_patterns_total; i++) {
		free(x1[i]);
		free(x[i]);
	}
	free(x1);
	free(x);
	for(i = 0; i < N_PERCEPTRONS; i++) {
		free(dik[i]);
	}
	free(dik);
	free(d);

}




void finish() {
	free_memory();
	evaluate_time();
	fclose(pf_results);
}


main() {
	inicializa();

	for(trial = 0; trial < N_TRIALS; trial++) {
		generate_patterns();
		evaluate_data();
		preprocessing();
		for(inic = accuracy_trial[trial] = 0; inic < N_INIC; inic++) {
			calculate_weights();
		}
		validate();
		printf("*****TRIAL %i: accuracy= %g%%\n", trial, accuracy_trial[trial]);
		fprintf(pf_results, "*****TRIAL %i: accuracy= %g%%\n", trial, accuracy_trial[trial]);
	}

	average_results();
	finish();
}
