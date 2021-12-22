from _typeshed import Self
import abc
import operator

from numpy.lib.index_tricks import fill_diagonal

import default_graph
import numpy as np
'''
计算图节点类
'''

#概念说明：
#jacobi 矩阵：shape=（m*n），m代表标量函数的个数，n代表权重变量的个数。矩阵中的数值代表某函数对某变量的梯度。

class Node(object):
    def __init__(self, *parents, **kwargs) -> None:
        super().__init__()
        ############################
        # 构建父子关系
        ############################
        self.parents = list(parents)
        self.children = []
        for parent in self.parents:
            parent.children.append(self)

        self.value = None #当前节点的值
        self.jacobi = None #当前节点的jacobi矩阵

        ############################
        # 维护全局计算图
        ############################
        self.graph = kwargs.get('graph', default_graph)
        self.graph.add_node(self)

        ############################
        # 与save有关
        ############################
        self.name = self.gen_node_name(**kwargs)
        self.need_save = kwargs.get('need_save', True)
        
    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self, **kwargs):
        #############################
        #生成节点名称，默认如：MatMul:3
        #############################
        name = kwargs.get('node_name', "{}.{}".format(self.__class__.__name__, self.graph.node_count))
        if self.graph.name_scope:
            name = "{}/{}".format(self.graph.name_scope, name)
        return name

    def forward(self):
        #####################################################
        #计算本节点的值，如果父节点未被计算，则先递归计算父节点
        #####################################################
        if self.parents:
            for parent in self.parents:
                if parent.value is None:
                    parent.forward()
        self.compute()

    def backward(self, result):
        if self.jacobi is None:
            if self == result:
                self.jacobi = np.eye(self.dimension)
            else:
                self.jacobi = np.zeros((result.dimension, self.dimension))#初始化梯度为0

                #累加所有有效路径上的子节点的梯度
                for child in self.children:
                    if child.value is not None:
                        self.jacobi += np.dot(child.backward(result), child.father_wants_jacobi(self)) # 一个是最终节点对子节点的梯度，一个是子节点对爸爸我的梯度
        return self.jacobi

    @property
    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]

    @property
    def shape(self):
        return self.value.shape

    def reset_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()
    
    def clear_jacobi(self):
        self.jacobi = None

    @abc.abstractmethod
    def compute(self):
        pass
    
    @abc.abstractmethod
    def father_wants_jacobi(self, parent):
        #####################################################
        #父节点计算自己的jacobi时调用子节点的此方法，并将自己传入
        #子节点根据表达式计算对传入的父节点的jacobi
        #####################################################
        pass
    

class Variable(Node):
    '''
    变量节点，该节点无父节点
    '''

    def __init__(self, dim, init=False, trainable=True,**kwargs) -> None:
        Node.__init__(self, **kwargs)
        self.dim = dim

        #默认使用正态分布初始化
        if init:
            self.value = np.random.normal(0, 0.001, self.dim)
        
        self.trainable = trainable

    def set_value(self, value):
        assert value.shape == self.dim
        self.reset_value()
        self.value = value

class Operator(Node):
    pass

class MatMul(Operator):
    def compute(self):
        #parent[0], left shape: m * n
        #parent[1], right shape: n * k
        assert len(self.parents) == 2 and self.parents[0].shape[1] == self.parents[1].shape[0]
        self.value = np.dot(self.parents[0].value, self.parents[1].value)#m * k
    
    def father_wants_jacobi(self, parent):
        jacobi = np.zeros((self.dimension, parent.dimension), dtype=float)#(m*k) * (m*n)或(m*k) * (n*k)
        if parent == self.parents[0]:
            return fill_diagonal(jacobi, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(jacobi, self.parents[0].value)
            row_sort = np.arange(self.dimension).reshape(
                self.shape[:: -1]).T.ravel()
            col_sort = np.arange(parent.dimension).reshape(
                parent.shape[::-1]).T.ravel()
        
            return jacobi[row_sort, :][:, col_sort]

            
            