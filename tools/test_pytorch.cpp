#include <Python.h>  
#include <iostream>  
#include <string>  
using std::cin;  
using std::cout;  
using std::endl;  
using std::string;  
  
int main(int argc, char **argv)  
{  
	if(argc < 3 )
    {
        cout<<"usage: detector.bin  imagename"<<endl;
		return -1;
    }
	
    string param_1(argv[1]);
	string param_2(argv[2]);
	
	int par_1 = atoi( param_1.c_str() );
	
	int par_2 = atoi( param_2.c_str() );


    Py_Initialize(); //初始化 python  
    if (!Py_IsInitialized())  
    {  
        cout << "initialized error" << endl;  
        return -1;  
    }  
  
    PyRun_SimpleString("import  sys"); // 执行 python 中的短语句  
	//PyRun_SimpleString("import  numpy as np"); // 执行 python 中的短语句  

    PyRun_SimpleString("sys.path.append('/media/media_share/linkfile/crnn_pytorch/')");  
	PyRun_SimpleString("sys.path.append('/media/media_share/linkfile/crnn_pytorch/models/')"); 
  
    PyObject *pName(0), *pModule(0), *pDct(0), *pFunc(0), *pArgs(0);  
  
    pName = PyString_FromString("demo"); //载入名为 pytest的脚本  
    pModule = PyImport_Import(pName);  
  
    if (!pModule)  
    {  
        cout << "can not find demo.py" << endl;  
        return -1;  
    }  
    else  
        cout << "open Module" << endl;  
  
    pDct = PyModule_GetDict(pModule);  
  
    if (!pDct)  
    {  
        cout << "pDct error" << endl;  
        return -1;  
    }  
  
    pFunc = 0;  
    pFunc = PyDict_GetItemString(pDct, "process"); //找到名为 add 的函数  
    if (!pFunc || !PyCallable_Check(pFunc))  
    {  
        cout << "pFunc error" << endl;  
        return -1;  
    }  
	
    //pArgs = PyTuple_New(2); //为传入形参开辟空间  
	pArgs = Py_BuildValue("(i, i)", 1, 2);
  
    //放置传入的形参，类型说明：  
    //s 字符串 , 均是C 风格的字符串  
    //i 整型  
    //f 浮点数  
    //o 表示一个 python 对象  
	
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", par_1));  
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", par_2));  
    PyObject_CallObject(pFunc,NULL); 
	Py_DECREF(pArgs);
	Py_DECREF(pFunc);
	Py_DECREF(pModule);  
    Py_Finalize(); 
	cout << "yes is ok" << endl;
	return 0;
	
}  

