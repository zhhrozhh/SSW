#include <iostream>
int DS[13] ={0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
class Date{

	int N;

	int SetDate(int N){
		this->N = N;
		return 0;
	}
	int GetDate(){
		return this->N;
	}
	void SetDate(int N){
		this->N = N;
	}
	int GetDate(int&Y,int&M,int&D){
		Y = (this->N / 365) + 1990;
		for(size_t i = 1;i < 12;i++){


		}
	}
	void SetDate(int Y,int M,int D){
		this -> N = (Y-1990)*365 + D + DS[M];
	}
}