BEGIN {
    n=0; 
    init=0;
} 
($1 !~/#/) { 
    ncol = NF; 
    if (init == 0) {
        for (j = 1; j <= ncol; j++) { 
            data_average[j] = 0;
        }
    };
    init = 1;
}
(init == 1 && $1 !~/#/) { 
    if (NF != ncol) { 
        print "format error"; 
        exit -1;
    } 
    for (j=1;j<=ncol;j++) {
        data[n,int((j+2)%6)+1] = $j; 
        data_average[int((j+2)%6)+1] = (data[n,int((j+2)%6+1)] + data_average[int((j+2)%6+1)]*n ) / (n+1);
    } 
    n++;
} 
END { 
    for (i = 0; i < n; i++) { 
        printf("%d", i+1); 
        for (j = 1;j <=ncol; j++) {
            printf ("%10.6f", data[i,j] - data_average[j]); 
        }
        printf("%10.6f\n", 1);
    }
}