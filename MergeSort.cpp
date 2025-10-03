#include<iostream>
using namespace std;

void printArr(int Arr[],int n){
    for(int i=0;i<n;i++)
        cout<<Arr[i]<<"\n";
}
void mergeArray(int Arr[], int s, int m, int e){
    int len1 = m-s+1;
    int len2 = e-m;
    int *tempArr1 = new int[len1];
    int *tempArr2 = new int[len2];
    int *NewArr = new int[len1+len2];

    for(int i=0; i<len1; i++)
        tempArr1[i] = Arr[s+i];

    for(int i=0; i<len2; i++)
        tempArr2[i] = Arr[m+1+i];

    int leftIndex = 0, rightIndex = 0, mergedIndex = 0;

    while(leftIndex<len1 && rightIndex<len2){
        if(tempArr1[leftIndex] < tempArr2[rightIndex]){
            NewArr[mergedIndex] = tempArr1[leftIndex];
            mergedIndex++;
            leftIndex++;
        }
        else if(tempArr1[leftIndex] > tempArr2[rightIndex]){
            NewArr[mergedIndex] = tempArr2[rightIndex];
            mergedIndex++;
            rightIndex++;
        }
        else{
            NewArr[mergedIndex] = tempArr1[leftIndex];
            mergedIndex++;
            leftIndex++;
            NewArr[mergedIndex] = tempArr2[rightIndex];
            mergedIndex++;
            rightIndex++;
        }
    }
    while(leftIndex<len1){
        NewArr[mergedIndex] = tempArr1[leftIndex];
        mergedIndex++;
        leftIndex++;
    }

    while(rightIndex<len2){
        NewArr[mergedIndex] = tempArr2[rightIndex];
        mergedIndex++;
        rightIndex++;
    }
    for(int i=s; i<=e; i++)
        Arr[i] = NewArr[i-s];

}
void mergeSort(int Arr[],int start, int endIndex){
    if(start>=endIndex)
        return;
    int mid = (start + endIndex) /2;
    mergeSort(Arr,start,mid);
    mergeSort(Arr,mid+1,endIndex);
    mergeArray(Arr,start, mid, endIndex);

}
int main(){
    int Arr[]={99,72,11,34, 56,62,28,89,98,1,10,39,999,1};
    int len = sizeof(Arr)/sizeof(int);
    mergeSort(Arr,0,len-1);
    printArr(Arr,len);
    return 0;
}
