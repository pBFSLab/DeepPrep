BEGIN {
    n=0; 
    init=0;
}
($1 !~/#/) { 
    ncol = NF; 
    if (init == 0) {
        printf ("%d",1); 
        for (j = 1; j <= ncol; j++) {
            printf("%10.6f", 0);
        }
        printf("%10.6f\n",1); 
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
    }
    n++;
} 
END { 
    for (i = 1; i < n; i++) { 
        printf("%d", i+1); 
        for (j = 1;j <=ncol; j++) {
            printf ("%10.6f", data[i,j] - data[i-1,j]);
        }
        printf("%10.6f\n", 0);
    }
}