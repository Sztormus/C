// Parallel Sorting By Regular Sampling
// Author: Piotr Stefaniak

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <float.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>

//Function for getting actual time
double get_clock()
{
    struct timeval tv;
    struct timezone tz;
    int time = gettimeofday(&tv, &tz);
    if (time < 0)
    {
        fprintf(stderr, "get_clock() error");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

//Function for comparing two numbers
int compare(const void *num1, const void *num2)
{
    unsigned long long *n1 = (unsigned long long *)num1;
    unsigned long long *n2 = (unsigned long long *)num2;
    return (*n1 > *n2) - (*n1 < *n2);
}

int main(int argc, char *argv[])
{
    int size, nbuckets, my_rank, num_proc;
    double time_start, time_end;
    unsigned long long *pivots, *tab, *samples, **buckets;
    unsigned long long *local_tab, *local_samples;
    int *bucket_sizes, *global_bucket_sizes;
    int *partitonLength, *partitionStart;

    //checking input
    if (argc != 2)
    {
        fprintf(stderr, "\tWrong number of arguments.\nUsage: %s N\n", argv[0]);
        return -1;
    }

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    nbuckets = num_proc;

    size = atoi(argv[1]);

    //checking input
    if(size%nbuckets != 0)
    {
        if(my_rank == 0)
            printf("The size must be divisible by the number of processes\n");
        MPI_Finalize();
        return -1;
    }

    if(size/nbuckets < 5)
    {
        if(my_rank == 0)
            printf("Size is too small compared to number of processes\n");
        MPI_Finalize();
        return -1;
    }

    if (my_rank == 0)
    {
        printf("\tParallel Sorting By Regular Sampling\n\tAuthor: Piotr Stefaniak\n");
        printf("\tNumber of processes: %d\n\n", nbuckets);
        tab = malloc(sizeof(unsigned long long) * size);
    }

    local_tab = malloc(sizeof(unsigned long long) * (size / nbuckets));
    local_samples = malloc(sizeof(unsigned long long) * (nbuckets));

    pivots = malloc(sizeof(unsigned long long) * (nbuckets - 1));
    samples = malloc(sizeof(unsigned long long) * (nbuckets * nbuckets));
    buckets = malloc(sizeof(unsigned long long *) * nbuckets);

    partitonLength = malloc(sizeof(int) * nbuckets);
    partitionStart = malloc(sizeof(int) * nbuckets);

    for (int i = 0; i < nbuckets; ++i)
        buckets[i] = (unsigned long long *)malloc(sizeof(unsigned long long) * 2 * size / nbuckets);

    bucket_sizes = (int *)malloc(sizeof(int) * nbuckets);
    for (int i = 0; i < nbuckets; ++i)
        bucket_sizes[i] = 0;

    global_bucket_sizes = (int *)malloc(sizeof(int) * (nbuckets * nbuckets));
    for (int i = 0; i < nbuckets * nbuckets; ++i)
        global_bucket_sizes[i] = 0;

    //creating random table and saving it to file
    if (my_rank == 0)
    {
        long long top_value;
        if (size < 1000)
            top_value = 1000;
        else
            top_value = size;
        FILE *f = fopen("input.txt", "wt");
        srand(time(NULL));
        for (int i = 0; i < size; ++i)
        {
            tab[i] = rand() % top_value;
            fprintf(f, "%d ", tab[i]);
        }
        fclose(f);

        printf("\tUnsorted array has been saved in input.txt\n\n");
    }

    time_start = get_clock();
    if (my_rank == 0)
        printf("\tCalculating...\n\n");

    if (nbuckets > 1)
    {
        //PHASE 1: divide tab between processes and select local samples
        MPI_Scatter(tab, size / nbuckets, MPI_UNSIGNED_LONG_LONG, local_tab, size / nbuckets, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        qsort(local_tab, size / nbuckets, sizeof(unsigned long long), compare);

        int step = size / (nbuckets * nbuckets);
        for (int j = 0; j < nbuckets; ++j)
        {
            local_samples[j] = local_tab[j * step];
        }

        MPI_Allgather(local_samples, nbuckets, MPI_UNSIGNED_LONG_LONG, samples, nbuckets, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

        //PHASE 2: select pivots from all samples
        qsort(samples, nbuckets * nbuckets, sizeof(unsigned long long), compare);

        for (int i = 1; i < nbuckets; ++i)
        {
            pivots[i - 1] = samples[i * nbuckets];
        }

        if (my_rank == 0)
        {
            printf("\tPivots: ");
            for (int i = 0; i < nbuckets - 1; ++i)
            {
                printf("%d ", pivots[i]);
            }
            printf("\n\n");
        }

        //PHASE 3: divide local tabs based on piviots and distribute results between processes
        int index = 0;
        for (int i = 0; i < nbuckets - 1; ++i)
        {
            for (int j = 0; j < size / nbuckets; ++j)
            {
                if (local_tab[index] < pivots[i])
                {
                    buckets[i][j] = local_tab[index];
                    ++index;
                }
                else
                {
                    bucket_sizes[i] = j;
                    break;
                }
            }
        }

        for (int j = 0; j < size / nbuckets; ++j)
        {
            if (index < size / nbuckets)
            {
                buckets[nbuckets - 1][j] = local_tab[index];
                ++index;
            }
            else
            {
                bucket_sizes[nbuckets - 1] = j;
                break;
            }
        }

        MPI_Allgather(bucket_sizes, nbuckets, MPI_INT, global_bucket_sizes, nbuckets, MPI_INT, MPI_COMM_WORLD);

        int local_size = 0;
        for (int i = my_rank; i < nbuckets * nbuckets; i += nbuckets)
            local_size += global_bucket_sizes[i];

        free(local_tab);
        local_tab = malloc(sizeof(unsigned long long) * local_size);

        for (int proc_index = 0; proc_index < nbuckets; ++proc_index)
        {
            partitionStart[0] = 0;

            for (int i = 0; i < nbuckets; ++i)
                partitonLength[i] = global_bucket_sizes[my_rank + i * nbuckets];

            for (int i = 1; i < nbuckets; ++i)
                partitionStart[i] = partitionStart[i - 1] + partitonLength[i - 1];

            MPI_Gatherv(buckets[proc_index], bucket_sizes[proc_index], MPI_UNSIGNED_LONG_LONG, local_tab, partitonLength, partitionStart, MPI_UNSIGNED_LONG_LONG, proc_index, MPI_COMM_WORLD);
        }

        //PHASE 4: sort processed local tabs and merge them in process 0
        qsort(local_tab, local_size, sizeof(unsigned long long), compare);

        partitionStart[0] = 0;
        for (int i = 0; i < nbuckets; ++i)
        {
            int temp_sum = 0;
            for (int j = 0; j < nbuckets; ++j)
                temp_sum += global_bucket_sizes[i + j * nbuckets];
            partitonLength[i] = temp_sum;
        }

        for (int i = 1; i < nbuckets; ++i)
            partitionStart[i] = partitionStart[i - 1] + partitonLength[i - 1];

        MPI_Gatherv(local_tab, local_size, MPI_UNSIGNED_LONG_LONG, tab, partitonLength, partitionStart, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }
    else
        qsort(tab, size, sizeof(unsigned long long), compare);

    //show time and save sorted array to file
    if (my_rank == 0)
    {
        time_end = get_clock();
        printf("\tTime: %lf\n\n", (time_end - time_start));

        FILE *f = fopen("output.txt", "wt");
        for (int i = 0; i < size; ++i)
            fprintf(f, "%d ", tab[i]);
        fclose(f);

        printf("\tSorted array has been saved in output.txt\n\n");
        printf("\tCleaning...\n");

        free(tab);
    }

    free(local_tab);
    free(local_samples);
    free(pivots);
    free(samples);
    free(bucket_sizes);
    free(global_bucket_sizes);
    free(partitionStart);
    free(partitonLength);

    for (int i = 0; i < nbuckets; ++i)
    {
        free(buckets[i]);
    }
    free(buckets);

    MPI_Finalize();
    return 0;
}