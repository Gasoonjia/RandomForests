#include"Tree.h"
#include <pthread.h>
#include <time.h>
using namespace std;
//#include <thread>

Tree::Tree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
{
	_MaxDepth=MaxDepth;
	_trainFeatureNumPerNode=trainFeatureNumPerNode;
	_minLeafSample=minLeafSample;
	_minInfoGain=minInfoGain;
	_nodeNum=static_cast<int>(pow(2.0,_MaxDepth)-1);
	_cartreeArray=new Node*[_nodeNum];
	_isRegression=isRegression;
	for(int i=0;i<_nodeNum;++i)
	{_cartreeArray[i]=NULL;}
}

Tree::~Tree()
{
	if(_cartreeArray!=NULL)
	{
		for(int i=0;i<_nodeNum;++i)
		{
			if(_cartreeArray[i]!=NULL)
			{
				delete _cartreeArray[i];
				_cartreeArray[i]=NULL;
			}
		}
		delete[] _cartreeArray;
		_cartreeArray=NULL;
	}
}

Result Tree::predict(float*data)
{
	int position=0;
	Node*head=_cartreeArray[position];
	while(!head->isLeaf())
	{
		position=head->predict(data,position);
		head=_cartreeArray[position];
	}
	Result r;
	head->getResult(r);
	return r;
}
/************************************************/
//Classification Tree
ClasTree::ClasTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
	:Tree(MaxDepth,trainFeatureNumPerNode,minLeafSample,minInfoGain,isRegression)
{}
ClasTree::~ClasTree()
{}

struct args_t{
	int i;
	Node** _cartreeArray;
	int _nodeNum;
	int _minLeafSample;
	Sample* sample;
	float _minInfoGain;
	int _trainFeatureNumPerNode;

	args_t(int i, Node** _cartreeArray, int _nodeNum, int _minLeafSample, Sample* sample, float _minInfoGain,
	int _trainFeatureNumPerNode){
		this->i = i;
		this->_cartreeArray = _cartreeArray;
		this->_nodeNum = _nodeNum;
		this->_minLeafSample = _minLeafSample;
		this->sample = sample;
		this->_minInfoGain = _minInfoGain;
		this->_trainFeatureNumPerNode = _trainFeatureNumPerNode;
	}
};

void *processNode(void *args){
	args_t* myargs = (args_t*)args;
	int*_featureIndex=new int[myargs->_trainFeatureNumPerNode];
	//printf("%d started\n", myargs->i);
	int parentId=(myargs->i-1)/2;
	//if current node's parent node is NULL,continue 
	if(myargs->_cartreeArray[parentId]==NULL)
	{return NULL;}
	//if the current node's parent node is a leaf,continue
	if(myargs->i>0&&myargs->_cartreeArray[parentId]->isLeaf())
	{
		//printf("%d is leaf\n", myargs->i);
		return NULL;
	}
	//if it reach the max depth
	//set current node as a leaf and continue
	if(myargs->i*2+1>=myargs->_nodeNum)  //if left child node is out of range
	{
		myargs->_cartreeArray[myargs->i]->createLeaf();
		return NULL;
	}
	//if current samples in this node is less than the threshold
	//set current node as a leaf and continue
	if(myargs->_cartreeArray[myargs->i]->_samples->getSelectedSampleNum()<=myargs->_minLeafSample)
	{
		myargs->_cartreeArray[myargs->i]->createLeaf();
		return NULL;
	}
	myargs->_cartreeArray[myargs->i]->_samples->randomSelectFeature
	(_featureIndex,myargs->sample->getFeatureNum(),myargs->_trainFeatureNumPerNode);
	//else calculate the information gain
	//printf("%d before calculate\n", myargs->i);
	myargs->_cartreeArray[myargs->i]->calculateInfoGain(myargs->_cartreeArray,myargs->i,myargs->_minInfoGain);
	myargs->_cartreeArray[myargs->i]->_samples->releaseSampleIndex();
	//printf("%d completed\n", myargs->i);
	delete[] _featureIndex;
	return NULL;
}

void ClasTree::train(Sample*sample)
{
	//initialize root node
	//random generate feature index
	int*_featureIndex=new int[_trainFeatureNumPerNode];
    Sample*nodeSample=new Sample(sample,0,sample->getSelectedSampleNum()-1);
	_cartreeArray[0]=new ClasNode();
	_cartreeArray[0]->_samples=nodeSample;
	//calculate the probablity and gini
	_cartreeArray[0]->calculateParams();


	for(int i=0; i<_nodeNum; i++){
		int parentId=(i-1)/2;
		//if current node's parent node is NULL,continue 
		if(_cartreeArray[parentId]==NULL)
		{continue;}
		//if the current node's parent node is a leaf,continue
		if(i>0&&_cartreeArray[parentId]->isLeaf())
		{continue;}
		//if it reach the max depth
		//set current node as a leaf and continue
		if(i*2+1>=_nodeNum)  //if left child node is out of range
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		//if current samples in this node is less than the threshold
		//set current node as a leaf and continue
		if(_cartreeArray[i]->_samples->getSelectedSampleNum()<=_minLeafSample)
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		_cartreeArray[i]->_samples->randomSelectFeature
		(_featureIndex,sample->getFeatureNum(),_trainFeatureNumPerNode);
		//else calculate the information gain
		_cartreeArray[i]->calculateInfoGain(_cartreeArray,i,_minInfoGain);
		_cartreeArray[i]->_samples->releaseSampleIndex();
	}

	// int i = 0;
	// int start = 0;
	// int levelLength = 1;

	// while(i < _nodeNum){
	// 	pthread_t* threads = (pthread_t*)malloc(levelLength*sizeof(pthread_t));
	// 	//printf("%d\n", levelLength);
	// 	args_t** args_list = new args_t*[levelLength];
	// 	for(i=start; i<start+levelLength && i<_nodeNum; ++i)
	// 	{
	// 		args_list[i-start] = new args_t(i, _cartreeArray, _nodeNum, _minLeafSample, sample,
	// 		_minInfoGain, _trainFeatureNumPerNode);
	// 		pthread_create(&threads[i-start], NULL, processNode, args_list[i-start]);
	// 		//pthread_join(threads[i-start], NULL);
	// 		//processNode(args_list[i-start]);
	// 	}
	// 	for(int j = 0; j<levelLength && start + j < _nodeNum; j++){
	// 		pthread_join(threads[j], NULL);
	// 		//printf("%d joined\n", start+j);
	// 		delete args_list[j];
	// 	}
	// 	delete[] args_list;
	// 	free(threads);
	// 	start = start+levelLength;
	// 	levelLength *= 2;
	// }



	delete[] _featureIndex;
	_featureIndex=NULL;
    delete nodeSample;
}

void ClasTree::setup(Sample *sample){
	//initialize root node
	//random generate feature index
    Sample*nodeSample=new Sample(sample,0,sample->getSelectedSampleNum()-1);
	_cartreeArray[0]=new ClasNode();
	_cartreeArray[0]->_samples=nodeSample;
	//calculate the probablity and gini
	_cartreeArray[0]->calculateParams();
}

void ClasTree::splitNode(Sample *sample, int nodeIdx){
	args_t arg(nodeIdx, _cartreeArray, _nodeNum, _minLeafSample, sample, _minInfoGain, _trainFeatureNumPerNode);
	processNode(&arg);
}

void ClasTree::createNode(int id,int featureIndex,float threshold)
{
	_cartreeArray[id]=new ClasNode();
	_cartreeArray[id]->setLeaf(false);
	_cartreeArray[id]->setFeatureIndex(featureIndex);
	_cartreeArray[id]->setThreshold(threshold);
}

void ClasTree::createLeaf(int id,float clas,float prob)
{
	_cartreeArray[id]=new ClasNode();
	_cartreeArray[id]->setLeaf(true);
	((ClasNode*)_cartreeArray[id])->setClass(clas);
	((ClasNode*)_cartreeArray[id])->setProb(prob);
}
/************************************************/
//Regression Tree
RegrTree::RegrTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
	:Tree(MaxDepth,trainFeatureNumPerNode,minLeafSample,minInfoGain,isRegression)
{}
RegrTree::~RegrTree()
{}
void RegrTree::train(Sample*sample)
{
	//initialize root node
	//random generate feature index
	int*_featureIndex=new int[_trainFeatureNumPerNode];
	Sample*nodeSample=new Sample(sample,0,sample->getSelectedSampleNum()-1);
	_cartreeArray[0]=new RegrNode();
	_cartreeArray[0]->_samples=nodeSample;
	//calculate the mean and variance
	_cartreeArray[0]->calculateParams();
	for(int i=0;i<_nodeNum;++i)
	{
		int parentId=(i-1)/2;
		//if current node's parent node is NULL,continue 
		if(_cartreeArray[parentId]==NULL)
		{continue;}
		//if the current node's parent node is a leaf,continue
		if(i>0&&_cartreeArray[parentId]->isLeaf())
		{continue;}
		//if it reach the max depth
		//set current node as a leaf and continue
		if(i*2+1>=_nodeNum)  //if left child node is out of range
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		//if current samples in this node is less than the threshold
		//set current node as a leaf and continue
		if(_cartreeArray[i]->_samples->getSelectedSampleNum()<=_minLeafSample)
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		_cartreeArray[i]->_samples->randomSelectFeature
		(_featureIndex,sample->getFeatureNum(),_trainFeatureNumPerNode);
		//else calculate the information gain
		_cartreeArray[i]->calculateInfoGain(_cartreeArray,i,_minInfoGain);
		_cartreeArray[i]->_samples->releaseSampleIndex();
	}
	delete[] _featureIndex;
	_featureIndex=NULL;
    delete nodeSample;
}

void RegrTree::createNode(int id,int featureIndex,float threshold)
{
	_cartreeArray[id]=new RegrNode();
	_cartreeArray[id]->setLeaf(false);
	_cartreeArray[id]->setFeatureIndex(featureIndex);
	_cartreeArray[id]->setThreshold(threshold);
}

void RegrTree::createLeaf(int id,float value)
{
	_cartreeArray[id]=new RegrNode();
	_cartreeArray[id]->setLeaf(true);
	((RegrNode*)_cartreeArray[id])->setValue(value);
}

void RegrTree::setup(Sample *sample){

}

void RegrTree::splitNode(Sample *sample, int nodeIdx){

}