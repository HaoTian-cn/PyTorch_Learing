'''
关于字符操作
name='tsy'
name.capitalize()为把第一个字符变成大写
name.upper()全变成大写
name.lower()全变成小写
name.strip()删除整个字符串中的空格
name.lstrip()删除左边空格
name.rstrip()删除右边空格
name.find('s')查找字符所在位置下标 没找到返回-1
name.index('s')也可以检测字符串中是否包含字符
name.endswith('y')返回逻辑值找最后一个字符是否对
name.startswith('t')返回逻辑值找第一个字符是否对
name.count('tsy')查找字符串出现从次数
name[0:2:1]从0元素找到1元素1为步长
'''

'''
关于列表操作
list1=[123,'magic',True]
lsit1[0]返回123
list1.append('888')追加一个’888‘在列表最后
list1.extend([123,'1230'])拓展列表到最后
list1.insert(1,'tsy')将’tsy‘插入列表的1号位置
del list1[0:2]批量删除多项数据
list1.remove('magic')移除指定数据项，每次只能移除一项
list1.pop(0)移除指定下标元素
list1.index('magic')找到对应元素位置索引
'''

'''
关于元组操作
元组创建之后无法做任何修改，具有不可变性，用小括号来创建元祖类型数据项也用逗号分割可以是任何类型
当元组中只有一个元素时要加上逗号不然解释器会当做整形处理，支持切片操作
tupleA=()
tupleA=('tsy',89,9.123,True,'peter',[1,'magic' ])初始化之后无法更改
tupleA[2:4]元祖的2，3号元素
tupleA[::-1]从右往左遍历数据
[start:end:step]起始点终止点步长
'''

'''
关于字典
可以存储任意对象以键值对的形式创建用{}创建
支持增删改查 特点：不是序列类型没有下标，无序的键值集合 高级数据类型，每个键值对用逗号分割，键必须是不可变的类型（字符串，元组）
键值可以是任意类型
dictA={}空字典
dictA['name']='tsy'
dictA['age']=19 添加字典
dictA['name'] 即可查找对应的值
dictA.keys() 获取所有的键
dictA.values() 获取所有的值
dictA.items() 获取所有的键值对
dictA.updata({'age':18}) 更新键值对，可以添加也可以修改
del dictA['name'] 删除指定键
dictA.pop('age') 删除指定键
排序 必须要类型一致
sorted(dictA.items(),key=lambda d:d[0])   d：d[0]代表按key排序，d：d[1]代表按value排序，key=lambda d:d[]是固定用法
'''
'''
公用操作
+ 两个对象相加操作会合并两个对象
* 复制 +几次
in 判断对象是否存在  
'''

'''
函数 参数有必选参数，默认参数，可选参数，关键字参数
可选参数用*args来表示，形参为元组，args为函数名
def c(*args):
    print(args)
c(1,2,3)
关键字参数用**args来表示，其形参形式为字典且key必须为字符串类型
def c(**args):
   print(args)
dictA={'name':'tsy','age':'18'}
c(**dictA) 第一种形式
c(name='tsy',age=18) 第二种形式
对于函数的定义必须是 必选参数，默认参数，可选参数，关键字参数的顺序 不能倒序使用
return 为返回值函数
'''

'''
内置sorted函数
all()其中一个是0 false 空就返回false
any()全为 0 false 空才返回false
sorted(iterable,cmp,key,reverse) sorted函数生成一个新的列表
iterable 可迭代对象
cmp比较函数这里必须有两个参数参数值从迭代对象中取出，规则为大于返回1小于返回-1等于返回0
key用来进行比较的元素只有一个参数，可以指定可迭代对象中的一个元素来排序
reverse 排序规则 reverse=True 降序 reverse=False 升序（默认）
返回值 重新排序的列表
reverse()反转列表函数
range（start，end，step)创建整数列表
zip() 相应索引的可迭代对象组合成两个元素的元组集合
enumerate()将可遍历的对象组合成一个索引序列，同时列出数据和数据下标**将对象打索引下标
enumerate()对字典进行标号时 封装的是key值
'''

'''
集合操作
{}创建集合 or set()
集合是无序且无法重复的 不支持切片和索引，只是一个容器
ste1={1,2,3}
ste1.add(4) 添加操作
set1.clear() 清空操作 在其他地方也都是可以的
set2={2,3,4}
set1.difference(ste2) 取ste1对set2的差集   set1-set2
set1.intersection(set2) 取交集   set1&set2
set1.union(set2) 取并集 set1|set2
set1.pop() 移除第一个并返回数据 
set1.discard(3) 指定移除3 变成{1,2}
set1.update({4,5,6}) 更新集合
'''

'''
面向对象编程 oop[object oriented programming]是一种python的编程思路
面向过程的不足，没有更多的精力去考虑别的事情，面向过程不适合做大项目
面向对象适合做大项目！！
面向过程的关注点是怎么做，面向对象的关注点的谁来做
oop以类的形式描述出来，以对象实例的新形式在软件系统中复用
类是一个模板，可以包含多个函数
对象是模板创建的实例，通过实例可以执行类中的函数
类：class 类名 类的属性：一组数据 类的方法：允许对进行操作的方法
定义类
class 类名:
   属性
   方法
创建对象
对象名=类名
'''

'''
实例方法
在类的内部 使用def定义的函数可以定义一个实例方法与一般的函数定义不同，类方法必须包含参数self且为第一个参数
不一定要self 可以用其他的名字但是这个位置必须要占用，归于类的实例所有
类属性定义在类里面，方法外面
定义在类方法里面使用self引用的属性称为实例属性就是对象的属性
实例属性：初始化
def __init__(self，，):
    self.name='小茗'
利用 传参的机制可以让我们定义功能强大并且方便的类
self 其实是实例对象本身，在调用时不必要传入参数
'''

'''
魔术方法
python 中内置好的特定方法__xxx__
__init__ 初始化一个类，创建实例对象时赋值
__str__ 将对象转换成字符串测试时候打印对象信息
__new__ 创建并返回一个实例对象，调用一次就得到一个对象
__class__ 获得已知对象的类
__del__ 对象在程序运行结束后进行对象销毁的时候调用这个方法

__new__函数是创建对象的函数 ，最先执行这个函数 return object.__new__(cls) 可以控制创建对象的一些属性限定
__new__ 至少有一个函数是cls 代表要实例化的类，此参数在实例化时有python解释器自动提供
'''

'''
类的继承、父类的调用、静态方法、继承和重写
当一个对象被删除或者销毁的时候会默认调用__del__方法 也称为析构方法 
__del__(self)是析构函数
del 对象  就是清空
'''
'''
在python中展现面向对象的 三大特征
封装 继承 多态
封装：把内容封装到某个地方，便于后面的使用 通过对象直接或者self来获取被封装的内容
继承： 和现实生活中的继承是一样的 子可以继承父的内容（属性或者行为） 子类有的父类不一定有
极大地提高效率，减少代码的重复编写，精简代码的层级结构 便于拓展
单继承:
class Animal: ....
class Dog(Animal):继承了Animal
多继承：可以继承多个类使用多个类的方法和属性
class D:....
class B(D):
class C(D):
class A(B.C):
在执行多个父类同方法时首先对A查找，再对B中查找 没有再去C里查找 注意不会去B的父类去查找 如果C类中没有 就会去D类中查找 广度优先 平级
如果 BC父类不同则深度优先 一直到B 的最后父类

在父类定义实例属性
class Dog:
    def __init__(self,name,color):
        self.name=name
        self.color=color
class Keji(Dog):
    def __init__(self,name,color):
        Dog.__init__(self,name,color)
     #   or super().__init__(name,color) 不用传入self
'''

'''
多态 多种状态、形态同一种行为
要满足两个条件：需要继承和重写 增加了拓展性
多态=继承+重写
class Animal:
    def say(self):
        print('我是一个动物')
class Duck(Animal):

    def say(self):
        print('我是一只鸭子') #重写父类方法
class Dog(Animal):
    def say(self):
        print('我是一只狗')
def common(obj):
    obj.say()
listObj=[Duck(),Dog()]
for i in listObj:
    common(i)
'''

'''
类属性可以被 类对象和实例对象共同访问使用的
实例属性只能由 实例对象使用
类方法 ：类对象所拥有的方法 要用@classmethod 标识
第一个参数必须是类对象  可以通过类对象和实例对象调用
静态方法 需要用@staticmethod 标识
不需要传参数  一般不用实例对象访问静态方法
为什么要使用静态方法，逻辑实现跟类和类对象没有交互，主要存放逻辑性的代码 不涉及类中方法和属性的操作
使数据资源得到充分有效的利用
'''

'''
私有化：对封装加锁
私有化属性 保护属性安全 
语法 两个下划线开头 声明该属性私有 不能被类外调用
只能在类里面被调用，外部无法使用，不能被继承
私有化属性可以在类里面被修改
私有化方法
语法 两个下划线开头 声明该方法私有 不能被类外调用
只能在类里面被调用，外部无法使用，不能被继承
property(get_age,set_age) 在类外调用私有化属性 以get set 开头
get 是得到私有化属性 set是设置私有化属性

或者使用装饰器添加属性:
class Person:
    def __init__(self):
        self.__name='张三' #加了俩下划线 将属性私有化
        self.age=30
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self,new_name):
        self.__name=new_name

'''

'''
__new__方法创建并返回一个实例对象
class A(object):
   def __new__(cls,*args,**kwargs)
   print('__new__方法执行')
   return super().__new__(cls,*args,**kwargs) # return object.__new__(cls,*args,**kwargs)
单例模式
单例模式确保某一个类只有一个实例存在
希望在整个系统中某个类智能出现一个实例的时候 那么这个单例对象就满足要求 节省数据库资源

'''


'''
异常处理

try:
   可能出现错误的代码块
except:
   出错后执行的代码块
except IndexError(错误类型) as a:
   ...
Exception 捕获所有异常 一次只能捕获一次错误
except Exception as error:
   ...
else:
   没有出错执行代码块
finally:
   不管有没有出错都执行的代码块
自定义异常
class errorname(Exception):
    def __init__(self,len):
        self.len=len
    def __str__(self):
        return '输入长度为{}超过了长度'.format(self.len)
def name_test():
    name=input('输入')
    try:
        if len(name)>4:
            raise errorname(len(name))
        else:
            print(name)
    except errorname as error:
        print(error)
name_test()
'''

'''
动态添加属性 方法
import  types #添加方法的库
class Animal:
    def cat_say(self):
        print('I can hang')
def name(self):
    print('我是猫')
cat=Animal()
cat.printname=types.MethodType(name,cat)
cat.printname()
__slots__变量 限制class 实例能添加的属性 只有在__slots__变量中的属性才能被添加
属性子类不会继承，只有在当前类中有效
class A:
   __slots__('name','age')
作用：减少内存空间不用字典形式储存于类当中
'''
'''
对象：我方飞机，敌方飞机，我方子弹，敌方子弹

功能:
添加背景音乐
控制我方飞机移动 敌方飞机随机移动
双方飞机都可以发射子弹

步骤：
1.创建一个窗口
2.创建我方飞机根据方向键 左右移动
3.给我方飞机添加发送子弹功能
4.创建敌方飞机
5.自由移动
6.发射子弹

'''
'''
打包 方式 最大限度减少 内存
新建文件夹 虚拟环境
win+R 盘符索引
pip install pipenv
pipenv install -python 3.6
pipenv shell
根据要打包的程序中导入的库，在 pipenv环境下 重新安装，例如：
pipenv install pyinstaller
pipenv install pygame
把py脚本文件复制到这个新建的目录下，重新运行 pyinstaller，方法、参数等同以往一样就OK。
pyinstaller -F -w game.py

'''
''''
文件定位
打开文件 /读写文件/保存文件/关闭文件
默认的是gbk中文编码，最好的习惯是打开文件的时候指定一个编码
fobj=open('文件路径','打开模式',encoding='utf-8')
以二进制形式去写数据
二进制形式可以写入视频和图片
fobj=open('文件路径','wb')
fobj.write('写入内容'.encode('utf-8'))
读取文件 
fobj=open('./test.txt','r',encoding='utf-8')
print(fobj.read(10)) 10代表数据量 ，不填就读取所有 会记录读了多少
print(fobj.readline()) 读一行 
print(fobj.readlines()) 把所有行作为单位作为列表对象
with 语句 不管是否发生异常 结束后都能关闭，上下文管理对象，优点 自动释放打开关联对象
with open('./test.txt','rb') as fobj:
    a=fobj.read()
    print(a.decode('utf-8'))
read r r+ rb rb+
r r+ 只读 适用普通读取场景
write w w+ wb+ wb a ab
w+ w wb 会创建文件 
二进制读写需要注意编码问题 默认是gbk
a ab a+ 会在原有的文件基础 文件指针末尾去添加

文件备份程序
def copyFile():
    oldfile =input('新文件的文件名')
    filelist=oldfile.split('.')
    newfile = filelist[0]+'备份'+filelist[1]
    old=open('./'+oldfile,'r',encoding='utf-8')
    new=open('./'+newfile,'w',encoding='utf-8')
    content=old.read()
    new.write(content)
    new.close()
    old.close()
copyFile()
文件定位
tell()指的是当前文件指针读取到的位置，光标位置
utf-8b编码一个英文占1个 一个汉字占3个
truncate 对源文件进行截取操作
with open('./test.txt','r+',encoding='utf-8') as f:
    print(f.truncate(3))
保留了前三个字符其他的删掉
定位其他位置
seek(offset,from) offset 偏移量单位字节 负数是往回偏移 正数是往前偏移
from位置 0表示文件开头，1表示当前位置 2 表示文件末尾
对于 r的模式打开文件 在文本文件中，没有使用二进制的形式打开文件 移动光标只允许从文件的开头计算相对位置
'''

'''
!!! axis=0  竖向的 1 横向的
import numpy as np
numpy 矩阵操作
array=np.array([[1,2,3],[4,5,6]])
print('number ogf dim',array.ndim)  矩阵的维度
print('shape:',array.shape)  矩阵的行数和列数以元组形式储存
print('size',array.size) 矩阵元素的个数
array=np.arange(10,19).reshape((3,3)) arange()等同range reshape((a,b))重构这个成为 a行b列的矩阵
array=np.linspace(1,10,20) 从1到10 生成20等段值
关于矩阵运算跟matlab一样 但是对于matlab中的.*和./就是*和/ 矩阵乘法就是 np.dot(a,b)  or a.dot(b) 矩阵除法是 
np.random.random((a,b))随机的a行b列  [0,1]
np.max 和 np.min 和np.sum 都是对整个矩阵进行运算
np.max(array,axis=1)对行 np.max(array,axis=0)对列 返回一个向量列表
np.argmax(array)找出最大值的索引返回的 是 在整个地方的最大索引 是将整个矩阵变成了一个向量进行索引 一行一行来
np.average(array) 返回平均值
np.median(array)找到中位数
np.cumsum(array) 累加
np.diff(array) 差分 后一个数减前一个数
np.nonzero(array) 输出每个坐标的索引，行号放一个数组，列号放一个数组
np.sort(array) 逐行排序 axis=0可以设置按列排序
array.T转置 or np.transpose(array)
np.clip(array,5,9) 表示截矩阵 所有小于5的数变成5 大于9 的变成9 中间不变
np.mean(array,axis=0) 按照列取平均值 跟average 其实是一个
索引array array[2][3] == array[2,3] 获取所有 array[2,:]
array.flatten()将矩阵转换成向量
np.vstack((a1,a2)) a1,a2矩阵上下合并
np.hstack((a1,a2)) 将a1 a2矩阵左右合并
np.concatenate((A,B,B,A),axis=0) 上下合并多个矩阵 axis=1 左右合并
array[:,np.newaxis] 将行向量变成列向量 or array.reshape(array.size,1)
np.split(array,2,axis=1) 纵向分割 成两块 axis=0 横向分割 如果进行不等分割就是np.array_split(array,3,axis=1)
np.vsplit(array,2) 纵向分割 np.hsplit(array,2) 横向分割
B只想要值但不想关联A
B=A.copy()
'''

'''
import pandas as pd
pd.Series([data],index=...) 创建pandas的序列 插入
下面这两句是对数据的每行每列拉一个名字出来
data=pd.date_range('20220907',periods=6)#描述行的索引
df=pd.DataFrame(np.random.randn(6,4),index=data,columns=['a','b','c','d'])
如果没有指定那么就是 0 1 2 3 ...这样子的默认作为他们的名字
df.describe() 运算数字型的数字帧描述
df.T 转置
df.sort_index(axis=1,ascending=False) 按照index排序 对列标的名称进行排序 False 的话是倒着排序 axis=0那么是对行标排序
df.sort_values(by='a') 按照value排序 根据'a'列排序
输出 df.a df[0:3] 0:3行 df['20220907':'20220910'] 简单选择行列
df.loc['20130910']以标签的名义来选择 横向的标签  df.loc[:,['a','b']] 整行或者整列数据
df.iloc[3,1] 位置索引 第四行第二列
df.a[df.b>0]=0 逻辑更改值
df.dropna(axis=0) 将nan存在的某一行删除
df.fillna(value=0) 将所有的nan数字填入0
df.isnull() 是否有确实数据 打印一个逻辑矩阵
pandas中导入导出数据
data=pd.read_csv()
pd.to_csv()
pandas合并数据
res=pd.concat([df1,df2,df3],axis=0) merge
如果需要把标号重新更改 则 res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
'''

'''
matplotlib.pyplot模块 画图
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-1,1,50)
y=x**2
plt.plot(x,y)
plt.show()
plt.plot(x,y,color='red',linewidth=1.0,linestyle='--') 各种参数
plt.figure() 图片

坐标轴设置
plt.xlim(-1,2) 坐标轴限制
plt.ylim(-2,3)
plt.xlabel('x=x')坐标轴名称
plt.ylabel('y=11')
plt.title('title') 名称
plt.xticks() 坐标轴上对应的标尺
plt.yticks([-2,-1.8,-1],[r'$bad\$',r'$good$',r'$well$']) r'$  $' 修改字体，\是空格

设置边框
ax=plt.gca() #四个脊梁边框  
ax.spines['right'].set_color('none') 边框消失
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') 改变下坐标轴的位置到-1
ax.spines['bottom'].set_position(('data',-1))

图例
在plot加label参数 
后面加上 plt.label()
plt.label(loc='best') 最好的位置

散点图
n=1024
x=np.random.normal(0,1,n)
y=np.random.normal(0,1,n)
t=np.arctan2(y,x) #颜色好看
plt.scatter(x,y,s=75,alpha=0.5,c=t)

柱状图
n=12
x=np.random.normal(0,1,n)
y=(1-x/float(n))*np.random.uniform(0.5,1,n)
plt.bar(x,y,facecolor='#CEFF7C',edgecolor='white')
plt.show()
下面这段语句是给他们加y值标注
for x,y in zip(x,y):
    plt.text(x,y,'%.2f'%y,ha='center',va='bottom')

等高线图
def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y) #网格的输入值
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot) #8是等高线的密集程度 等高线设置
C=plt.contour(X,Y,f(X,Y),8,colors='black') #等高线 输出
plt.clabel(C,inline=True) #标注数字描述在等高线里面
plt.show()

图像
plt.imread() 
plt.imshow()

3d图
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
x=np.arange(-4,4,0.25)
y=np.arange(-4,4,0.25)
x,y=np.meshgrid(x,y)
r=np.sqrt(x**2+y**2)
z=np.sin(r)
ax.plot_surface(x,y,z,cmap=plt.get_cmap('rainbow'),rstride=1,cstride=1) #彩虹cmap  rstride=2,cstride=2 c是跨度
ax.contourf(x,y,z,zdir='z',offset=-2) #zdir 从哪个轴压下去映射 offset 设置在轴的哪个位置上映射
plt.show()

subplot 分区
plt.figure()
plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
plt.show()
跟matlab一样

次坐标轴
x=np.arange(0,10,0.1)
y1=0.05*x**2
y2=-1*y1
ax1=plt.gca()
ax2=ax1.twinx()
ax1.plot(x,y1,'g-')
ax2.plot(x,y2)
plt.show()
'''

'初始化方式'
'梯度学习率的设置'
'loss的改进'
'acc和loss的曲线'
'混淆矩阵'
'tsne特征可视化'
'参加比赛，科研'
'''
https://blog.csdn.net/leiduifan6944/article/details/105075576?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166469604716782248589542%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166469604716782248589542&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-4-105075576-null-null.142^v51^pc_rank_34_2,201^v3^add_ask&utm_term=ARC%20loss&spm=1018.2226.3001.4187
https://blog.csdn.net/duan19920101/article/details/104445423?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166469601016800182754737%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166469601016800182754737&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-104445423-null-null.142^v51^pc_rank_34_2,201^v3^add_ask&utm_term=center%20loss&spm=1018.2226.3001.4187
https://blog.csdn.net/u012704941/article/details/82378345?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166469582716782391898487%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166469582716782391898487&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-82378345-null-null.142^v51^pc_rank_34_2,201^v3^add_ask&utm_term=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E5%BC%8F&spm=1018.2226.3001.4187
'''