//
// Created by sianh on 2025/1/7.
//

#include "sjtime.h"

#define NUM_LOOP 10000000

int *twoSum_v1(int *nums, int numsSize, int target, int *returnSize) {
    int *tab = (int *)malloc(2*sizeof(int));
    *returnSize = 2;
    // Hash table simulation
    for (int i = 0; i < numsSize; i++) {
        for (int j = i + 1; j < numsSize; j++) {
            if (nums[i] + nums[j]==target) {
                tab[0] = i;
                tab[1] = j;
                return tab;
            }
        }
    }
    return tab;
}

int *twoSum_v2(int *nums, int numsSize, int target, int *returnSize) {
    int i = 0;
    int j = 1;
    int a = numsSize;
    int *tab = (int *)malloc(sizeof(int)*2);
    *returnSize = 2;

    for (; numsSize > 0; numsSize--) {
        if (a - 1==i) {
            numsSize = a;
            i = 0;
            j++;
        }
        if (i + j < a) {
            if (nums[i] + nums[i + j]==target) {
                tab[0] = i;
                tab[1] = i + j;
                break;
            }
        }
        i++;
    }
    return tab;
}

int main() {
    int *result, return_size;
    int *nums = new int[4];
    nums[0] = 11;
    nums[1] = 15;
    nums[2] = 2;
    nums[3] = 7;

    sj::MultiWatch mw;
    mw.Insert("v1");
    mw.Insert("v2");

    mw["v1"].start();
    for (int i = 0; i < NUM_LOOP; ++i) {
        result = twoSum_v1(nums, 4, 9, &return_size);
    }
    mw["v1"].stop();

    mw["v2"].start();
    for (int i = 0; i < NUM_LOOP; ++i) {
        result = twoSum_v2(nums, 4, 9, &return_size);
    }
    mw["v2"].stop();

    printf("v1 running time (%d times) = %f(s)\n", NUM_LOOP, mw["v1"].duration());
    printf("v2 running time (%d times) = %f(s)\n", NUM_LOOP, mw["v2"].duration());

    delete[]nums;
}