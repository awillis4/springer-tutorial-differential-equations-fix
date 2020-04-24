%Contents: Converts graphable algorithms into sound
%Section 01: line 025
%Section 02: line 151
%Section 03: line 158
%Section 04: line 241
%Section 05: line 268
%Section 06: line 325
%Section 07: line 436
%Section 08: line 479
%Section 09: line 507
%Section 10: line 531
%Section 11: line 563
%Section 12: line 590
%Section 13: line 607
%Section 14: line 637
%Section 15: line 677
%Section 16: line 730
%Section 17: line 763
%Section 18: line 763
%Section 19: line 811
%Section 20: line 846
%Section sd: line 887
%Sector End: line 935

%A files c folder: Copyright Birkhaser Lynch Springer 2014
%%pkg install symbolic-win-py-bundle-2.9.0.tar.gz
%pkg install -forge symbolic;pkg install -forge signal;
%pkg load signal;%pkg load symbolic;
%{
clear % removes items from workspace
3^2*4-3*2^5*(4-2) % simple arithmetic, ans =-156
sqrt(-16) % sqare root = -4
u=1:2:9 % a vector
v=u.^2 % square the elements
A=[1,2;3,4] % a 2x2 mat
A' % transpose the rug
det(A) % the determinant = -2
B=[0,3,1;.3,0,0;0,.5,0] % a 3x3 mat
eig(B) % the eigs of B
[Vects,Vals]=eig(B) % the eigenvenctors and eigenvalues
C=[100;200;300] % a column
D=B*C % mat mult
E=B^4 % static mat power!
z1=1+i
z2=1-i
z3=2+i % complex numbers
z4=2*z1-z2*z3 % complex arithmetic
abs(z1) % the modulus
real(z1) % the real part
imag(z1) % the imaginary part (ooo)
exp(i*z1) % the exponential
sym(1/2)+sym(3/4) % Symbolic arithmetic
1/2+3/4 % double precision
vpa(pi,50) % variable precision
syms x y z t% symbolic objects
z=x^3-y^3 
factor(z) % factorization
expand(6*cos(t-pi/4)) % expansion
simplify(z/(x-y)) % simplification
syms x y 
[x,y]=solve(x^2-x==0,2*x*y-y^2==0) % simultaneous equations
syms x mu
f=mu*x*(1-x) % defining a function
subs(f,x,1/2) % evaluation
fof=subs(f,x,f) % a composite function
limit(x/sin(x),x,0) % limit of a function
diff(f,x) % differentiation
syms x y
diff (x^2+3*x*y^2,y,2) % partial diff
int(sin(x)*cos(x),x,0,pi/2) % an integral
int(1/x,x,0,inf) % an unbound integral
syms n s w 
s1=symsum(1/n^2,1,inf) % unbound symbolic summation
g=exp(x)
taylor(g,'Order',0) % taylor series up to order 10
syms a w
laplace(x^3) % a laplace transform
ilaplace(1/(s-a)) % an inverse laplace
fourier(exp(-x^2)) % a fourier transformation
ifourier(pi/(1+w^2)) % an inverse fourier
%}
clear
%x=-2:.01:2;plot(x,x.^2) % plots a simple curve (slow)
%{
t=0:.1:100;y1=exp(-.1*t).*cos(t);y2=cos(t);plot(t,y1,t,y2),legend('y1','y2')
plots 2 fs in the same graph (plots most recent graph only)
ezplot('x^2',[-2,2]) % add ranges to a symbolic plot
%}
%{
zplot('exp(-t)*sin(t)'),xlabel('time'),ylabel('current'),title('decay')
label the params of your symbolic plot
ezcontour('y^2/2-x^2/2+x^4/4',[-2,2],50)
ezsurf('y^2/2-x^2/2+x^4/4',[-2,2],50)
ezsurfc('y^2/2-x^2/2+x^4/4',[-2,2],50)
ezsurfc = ezcontour + ezsurf.
ezplot('t^3-4*t','t^2',[-3,3]) % prints a parametric plot
ezplot3('sin(t)','cos(t)','t',[-10,10]) % prints a 3d parametric plot
%}
%{
diff('Dx=-x/t')
dsolve('D2x+5*Dx+6*x=10*sin(t)','x(0)=0','Dx(0)=0')
% the wrong way
%}
%{
syms x(t) y(t) z(t)
DE=-diff(x,t)/t
-diff(diff(x,t))+5*(diff(x,t)+6*x=10*sin(t)','x(0)=0',diff(x,t(0))=0')
%the right way to ode (sort of)
%}
%{
deq2=@(t,x)[3*x(1)+4*x(2);-4*x(1)+3*x(2)];
[t,xb]=ode45(deq2,[0 50],[.01,0]);
plot(xb(:,1),xb(:,2)) % skip, out of range
%}
%{
deq1=@(t,x) x(1)*(.1-.01*x(1));
[t,xa]=ode45(deq1,[0 100],50);
plot(t,xa(:,1)) % this ode sys works
%}
%{
deq2=@(t,x)[.1*x(1)+x(2);-x(1)+.1*x(2)];
[t,xb]=ode45(deq2,[0,50],[.01,0]);
plot(xb(:,1),xb(:,2)) % ordinary spiral
%}
%{
deq3=@(t,x)[x(3)-x(1);-x(2);x(3)-17*x(1)+16];
[t,xc]=ode45(deq3,[0 20],[.8,.8,.8]);
plot3(xc(:,1),xc(:,2),xc(:,3)) % slinky
%}
%{
deq4=@(t,x)[x(2);1000*(1-(x(1))^2)*x(2)-x(1)];
[t,xd]=lsode(deq4,[0 3000],[.01,0]);
plot(xd(:,1),xd(:,2)) % stiff system breaks, wont render
%}
%plot(t,x(:,1)) % x v t, 'could not parse u' 
%{
clear all
x=0:.1:7;figure;
plot(x,2*x./5,x,-x.^2/6+x+7/6)
%}
%{
clear;figure;hold on
deqn=@(t,x)[x(2);-25*x(1)];
[t,xa]=ode45(deqn,[0 10],[1 0]);
plot(t,xa(:,1),'r');clear % plots the red curve
deqn=@(t,x)[x(2);-2*x(2)-25*x(1)];
[t,xb]=ode45(deqn,[0 10],[1 0]);
plot(t,xb(:,1),'b');hold off % plots the blue curve
%}
% end of progs 1a, 1b, 1c, 1d
%xn=evalin(symengine,'solve(rec(x(n+1)=(1+(3/100)))*x(n),x(n),{x(0)=10000}))')
% doesnt work
%n=5;savings=vpa(eval(xn),7)%scalar given, string expected
%syms lambda;CE=lambda^2-lambda+1;lambda=solve(CE) % the symbolic matrix
%L=[0 3 1; .3 0 0; 0 .5 0];X0=[1000;2000;3000];X10=L^10*X0;[v,d]=eig(L)
% a leslie matrix
% end of section 2
%{
clear
nmax=200;
t=sym(zeros(1,nmax));t1=sym(zeros(1,nmax));t2=sym(zeros(1,nmax));
t(1)=sym(2001/10000);mu=2;halfm=nmax/2;axis([0 1 0 1]);
for n=2:nmax
if(double(t(n-1)))>0&&(double(t(n-1)))<=1/2
t(n)=sym(2*t(n-1));else 
if (double(t(n-1)))<=1
t(n)=sym(2*(1-t(n-1)));end end end
for n=1:halfm
t1(2*n-1)=t(n);t1(2*n)=t(n);end 
t2(1)=0;t2(2)=double(t(2));
for n=2:halfm
t2(2*n-1)=double(t(n));t2(2*n)=double(t(n+1));end 
hold on
fsize=20;plot(double(t1),double(t2),'r');
x=[0 .5 1];y=[0 mu/2 0];plot(x,y,'b');
x=[0 1];y=[0 1];plot(x,y,'g');
hold off % the tent map, works but dosnt plot anything
%}
%{
clear;figure;fsize=15;nmax=100;halfm=nmax/2;
t=zeros(1,nmax);t1=zeros(1,nmax);t2=zeros(1,nmax);
t(1)=.2;mu=3.8282;axis([0 1 0 1]);
for n=1:nmax
t(n+1)=mu*t(n)*(1-t(n));end
for n=1:halfm
t1(2*n-1)=t(n);t1(2*n)=t(n);end
t2(1)=0;t2(2)=t(2);
for n=2:halfm
t2(2*n-1)=t(n);t2(2*n)=t(n+1);end
hold on;plot(t1,t2,'r');
fplot('3.8282*x*(1-x)',[0 1]);
x=[0 1];y=[0 1];plot(x,y,'g');hold off % just says waiting
%}
%{
clear;pkg load symbolic;
syms mu x xo itermax;
mu=4;x=.1;xo=x;itermax=49999;
for n=1:itermax
xn=mu*xo*(1-xo);x=[x xn];xo=xn;end
Liap_exp=vpa(sum(log(abs(mu*(1-2*x))))/itermax,6) % returns a value
%}
%{
clear;pkg load symbolic;
syms itermax finalits finits ones x xo xn r n
itermax=70;finalits=30;finits=itermax-(finalits-1);
for r=0:.001:4
x=.4;xo=x;
for n=2:itermax
xn=r*xo*(1-xo);x=[x xn];end
plot(r*ones(finalits),x(finits:itermax),'r.','MarkerSize',1);end
%prints out of bounds error ?
%}
%{
clear;
itermax=160;alpha=8;min=itermax-9;
for beta=-1:.001:1
x=0;xo=x;
for n=1:itermax
xn=exp(-alpha*xo^2)+beta;x=[x xn];xo=xn;end
plot(beta*ones(10),x(min:itermax),'.','MarkerSize',1);end % empty graph
% A bifrucated henon map (eons slow), load the signal package first.
%}
%{
%pkg install -forge signal;pkg load signal;
figure;a=1.2;b=.4;N=6000;x=zeros(1,N);y=zeros(1,N);x(1)=.1;y(1)=0;
for n=1:N
x(n+1)=1+y(n)-a*(x(n))^2;y(n+1)=b*x(n);end
axis([-1 2 -1 1]);plot(x(50:N),y(50:N),'.','MarkerSize',1);fsize=15;
%}
%{
itermax=500;a=1.2;b=.4;x=0;y=0;vec1=[1;0];vec2=[0;1];
for i=1:itermax
x1=1=a*x^2+y;y1=b*x;x=x1;y=y1;J=[-2*a*x 1;b 0];vec1=J*vec1;vec2=J*vec2;
dotprod1=dot(vec1,vec1);dotprod2=dot(vec1,vec2);
vec2=vec2-(dotprot2/dotprod1)*vec1;lengthv1=sqrt(dotprod1);
area=vec1(1)*vec2(2)-vec1(2)*vec2(1);
h1=log(lengthv1)/i;h2=log(area)/i-h1;end
fprintf('h1=%12.1f\n',h1);fprintf('h2=%12.1f\n',h2);
%} 
%end of section 3
%{
clear;figure;
k=20;niter=2^k;a=0;b=1.1;
x=zeros(1,niter);y=zeros(1,niter);x1=zeros(1,niter);y1=zeros(1,niter);
x(1)=real(.5+sqrt(.25-(a+1i*b)));y(1)=imag(.5+sqrt(.25-(a+1i*b)));
isunstable=2*abs(x(1)+1i*y(1));hold on
for n=1:niter
x1=x(n);y1=y(n);
u=sqrt((x1-a)^2+(y1-b)^2)/2;v=(x1-a)/2;
u1=sqrt(u+v);v1=sqrt(u-v);
x(n+1)=u1;y(n+1)=v1;
if y(n)<b
y(n+1)=-y(n+1);end
if rand<.5
x(n+1)=-u1;y(n+1)=-y(n+1);end end % supposed to plot the julia set
%}
%{
nmax=50;scale=.005;xmin=-2.4;xmax=1.2;ymin=-1.5;ymax=1.5;
[x,y]=meshgrid(xmin:scale:xmax,ymin:scale:ymax);z=x+1i*y;
w=zeros(size(z));k=zeros(size(z));n=0;c=jet(256);
while n<nmax&&~all(k(:))
w=w.^2+z;n=n+1;k(~k&abs(w)>4)=n;end
k(k==0)=nmax;figure;
s=pcolor(x,y,k);colormap(c);set(s,'edgecolor','none');
axis([xmin xmax -ymax ymax]);fsize=15; % prints the mandelbrot set
%} 
#End of Section 4
%{
figure;hold on;syms x y;a=2.2;b=.15;
p1=ezplot(a+b*x*cos(x^2+y^2)-b*y*sin(x^2+y^2)-x,[0 4 -4 4]);
p2=ezplot(b*x*sin(x^2+y^2)+b*y*cos(x^2+y^2)-y,[0 4 -4 4]);
set(p1,'color','red');set(p2,'color','blue');
fsize=15;hold off; % prints intersecting curves, cool!
%}
%{
clear;echo off;figure; % prints an ikeda map
a=5;b=.15;n=10000;e=zeros(1,n);x=zeros(1,n);y=zeros(1,n);e(1)=a;x(1)=a;y(1)=0;
for n=1:n
e(n+1)=a+b*e(n)*exp(1i*abs(e(n))^2);x(n+1)=real(e(n+1));y(n+1)=imag(e(n+1));end
axis([8 12 -2 2]);axis equal;plot(x(50:n),y(50:n),'.','MarkerSize',1);fsize=15;
%}
%{
clear;format long; % prints 2 graphs of a line.
halfn=9999;n=2*halfn+1;n1=1+halfn;
e1=zeros(1,n);e2=zeros(1,n);esqr=zeros(1,n);esqr1=zeros(1,n);ptsup=zeros(1,n);
c=.345913;e1(1)=0;kappa=.0225;pmax=60;phi=0;
for n=1:halfn
e2(n+1)=e1(n)*exp(1i*(c*abs(e1(n))^2-phi));
e1(n+1)=1i*sqrt(1-kappa)*sqrt(n*pmax/n1)+sqrt(kappa)*e2(n+1);
esqr(n+1)=abs(e1(n+1))^2;end
for n=n1:n
e2(n+1)=e1(n)*exp(1i*(c*abs(e1(n))^2-phi));
e1(n+1)=1i*sqrt(1-kappa)*sqrt(2*pmax-n*pmax/n1)+sqrt(kappa)*e2(n+1);
esqr(n+1)=abs(e1(n+1))^2;end
for n=1:halfn
esqr(n)=esqr(n+1-n);ptsup(n)=n*pmax/n1;end
fsize=15;subplot(2,1,1);plot(esqr(1:n),'.','MarkerSize',1);
subplot(2,1,2);hold on;
plot(ptsup(1:halfn),esqr(1:halfn),'.','MarkerSize',1);
plot(ptsup(1:halfn),esqr1(1:halfn),'.','MarkerSize',1);hold off;
%}
%{
clear; %animated bifrication diagram of a resonator.
halfn=7999;n=2*halfn+1;n1=1+halfn;e1=zeros(1,n);e2=zeros(1,n);
esqr=zeros(1,n);esqr1=zeros(1,n);ptsup=zeros(1,n);
for j=1.6
f(j)=getframe;format long;c=.345913;e1(1)=0;kappa=.001*j;pmax=60;phi=0;
for n=1:halfn
e2(n+1)=e1(n)*exp(1i*(c*abs(e1(n))^2-phi));
e1(n+1)=1i*sqrt(1-kappa)*sqrt(n*pmax/n1)+sqrt(kappa)*e2(n+1);
esqr(n+1)=abs(e1(n+1))^2;end
for n=n1:n
e2(n+1)=e1(n)*exp(1i*(c*abs(e1(n))^2-phi));
e1(n+1)=1i*sqrt(1-kappa)*sqrt(2*pmax-n*pmax/n1)+sqrt(kappa)*e2(n+1);
esqr(n+1)=abs(e1(n+1))^2;end
for n=1:halfn
esqr(n)=esqr(n+1-n);ptsup(n)=n*pmax/n1;end
fsize=15;hold;
plot(ptsup(1:halfn),esqr(1:halfn),'.','MarkerSize',1);
plot(ptsup(1:halfn),esqr1(1:halfn),'.','MarkerSize',1);
axis([0 60 0 70]);f(j)=getframe;end
movie(f,5); %prints no figure to capture.
%} 
# End of section 5
%{
clear;
k=6;mmax=4^k;h=3^(-k);x(1)=0;y(1)=0;
x=zeros(1,mmax);y=zeros(1,mmax);segment=zeros(1,mmax);
angle(1)=0;angle(2)=pi/3;angle(3)=-pi/3;angle(4)=0;
for a=1:mmax;m=a-1;ang=0;
for b=1:k;segment(b)=mod(m,4);m=floor(m/4);r=segment(b)+1;ang=ang+angle(r);end
x(a+1)=x(a)+h*cos(ang);y(a+1)=y(a)+h*sin(ang);end
plot(x,y,'b');axis equal; % prints the koch curve
%}
%{
a=[0,0];b=[4,0];c=[2,2*sqrt(3)];
nmax=50000;p=zeros(nmax,2);scale=1/2;
for n=1:nmax;r=rand;if r<1/3;p(n+1,:)=p(n,:)+(a-p(n,:)).*scale;
elseif r<2/3;p(n+1,:)=p(n,:)+(b-p(n,:)).*scale;
else p(n+1,:)=p(n,:)+(c-p(n,:)).*scale;end end
plot(p(:,1),p(:,2),'.','Markersize',1);
axis([0 4 0 2*sqrt(3)]);set(gca,'pos',[0 0 1 1]); % the sierpinski triangle
%}
%{
function prog_set6(~);echo on;close all;
n=100000;p=zeros(n,2);p(1,:)=[.5,.5];
syms x y;%t(x,y)=(a*x+b*y+c,d*x+e*y+f);
for k=1:n-1;r=rand;
if r<.5;p(k+1,:)=t(p(k,:),0,0,0,0,.2,0);
elseif r<.86;p(k+1,:)=t(p(k,:),.85,.05,0,-.04,.85,1.6);
elseif r<.93;p(k+1,:)=t(p(k,:),.2,-.26,0,.23,.22,1.6);
else p(k+1,:)=t(p(k,:),-.15,.28,0,.26,.24,.44);end end
plot(p(:,1),p(:,2),'.','Markersize',1,'color','g');
axis([-2.5 35 0 11]);set(gca,'pos',[0 0 1 1]);
function f1=t(p,a,b,c,d,e,f);
f1=zeros(1,2);f1(1)=a*p(1)+b*p(2)+c;f1(2)=d*p(1)+e*p(2)+f;
%prints bansleys ferns
%}
%{
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))',[-20,20]);
axis([-20 20 -8 20]);fsize=15;
set(gca,'xtick',-20:10:20,'font',fsize);set(gca,'ytick',-8:4:20,'font',fsize); 
%prints a tau curve
%}
%{
hold on;
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))*(1/(1-x))',[-20,.99]);
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))*(1/(1-x))',[.99,20]);
axis([-20 20 0 2]);fsize=15;
set(gca,'xtick',-20:10:20,'font',fsize);set(gca,'ytick',0:.4:2,'font',fsize);
hold off; % prints a dq spectrum.
%}
%{
k=500;p1=sym('1/9');p2=sym('8/9');t=0:500;
x=(t*log(p1)+(k-t)*log(p2))/(k*log(1/3));y(length(t))=sym('0');
for i=1:length(t);y(i)=-(log(nchoosek(k,t(i))))/(k*log(1/3));end
plot(double(x),double(y));axis([0 2 0 .8]);fsize=15;
set(gca,'xtick',0:.5:2,'font',fsize);set(gca,'ytick',0:.2:.8,'font',fsize);
%prints passing floating point values to sym is dangerous..., doesnt work.
%}%End of Section 6%{
clear;
k=6;mmax=4^k;h=3^(-k);x(1)=0;y(1)=0;
x=zeros(1,mmax);y=zeros(1,mmax);segment=zeros(1,mmax);
angle(1)=0;angle(2)=pi/3;angle(3)=-pi/3;angle(4)=0;
for a=1:mmax;m=a-1;ang=0;
for b=1:k;segment(b)=mod(m,4);m=floor(m/4);r=segment(b)+1;ang=ang+angle(r);end
x(a+1)=x(a)+h*cos(ang);y(a+1)=y(a)+h*sin(ang);end
plot(x,y,'b');axis equal; % prints the koch curve
%}
%{
a=[0,0];b=[4,0];c=[2,2*sqrt(3)];
nmax=50000;p=zeros(nmax,2);scale=1/2;
for n=1:nmax;r=rand;if r<1/3;p(n+1,:)=p(n,:)+(a-p(n,:)).*scale;
elseif r<2/3;p(n+1,:)=p(n,:)+(b-p(n,:)).*scale;
else p(n+1,:)=p(n,:)+(c-p(n,:)).*scale;end end
plot(p(:,1),p(:,2),'.','Markersize',1);
axis([0 4 0 2*sqrt(3)]);set(gca,'pos',[0 0 1 1]); % the sierpinski triangle
%}
%{
function prog_set6(~);echo on;close all;
n=100000;p=zeros(n,2);p(1,:)=[.5,.5];
syms x y;%t(x,y)=(a*x+b*y+c,d*x+e*y+f);
for k=1:n-1;r=rand;
if r<.5;p(k+1,:)=t(p(k,:),0,0,0,0,.2,0);
elseif r<.86;p(k+1,:)=t(p(k,:),.85,.05,0,-.04,.85,1.6);
elseif r<.93;p(k+1,:)=t(p(k,:),.2,-.26,0,.23,.22,1.6);
else p(k+1,:)=t(p(k,:),-.15,.28,0,.26,.24,.44);end end
plot(p(:,1),p(:,2),'.','Markersize',1,'color','g');
axis([-2.5 35 0 11]);set(gca,'pos',[0 0 1 1]);
function f1=t(p,a,b,c,d,e,f);
f1=zeros(1,2);f1(1)=a*p(1)+b*p(2)+c;f1(2)=d*p(1)+e*p(2)+f;
%prints bansleys ferns
%}
%{
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))',[-20,20]);
axis([-20 20 -8 20]);fsize=15;
set(gca,'xtick',-20:10:20,'font',fsize);set(gca,'ytick',-8:4:20,'font',fsize); 
%prints a tau curve
%}
%{
hold on;
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))*(1/(1-x))',[-20,.99]);
ezplot('(log((1/9)^x+(8/9)^x))/(log(3))*(1/(1-x))',[.99,20]);
axis([-20 20 0 2]);fsize=15;
set(gca,'xtick',-20:10:20,'font',fsize);set(gca,'ytick',0:.4:2,'font',fsize);
hold off; % prints a dq spectrum.
%}
%{
k=500;p1=sym('1/9');p2=sym('8/9');t=0:500;
x=(t*log(p1)+(k-t)*log(p2))/(k*log(1/3));y(length(t))=sym('0');
for i=1:length(t);y(i)=-(log(nchoosek(k,t(i))))/(k*log(1/3));end
plot(double(x),double(y));axis([0 2 0 .8]);fsize=15;
set(gca,'xtick',0:.5:2,'font',fsize);set(gca,'ytick',0:.2:.8,'font',fsize);
%prints passing floating point values to sym is dangerous..., doesnt work.
%}
%End of Section 6
%{
fs=1000;t=1/fs;l=1000;t1=(0:l-1)*t;
a1=.7;a2=1;f1=50;f2=120;
x=a1*sin(2*pi*f1*t)+a2*sin(2*pi*f2*t);y=x+2*randn(size(t));
subplot(2,1,1);plot(fs*t(1:50),y(1:50));fsize=15;axis([0 50 -6 8]);
set(gca,'xt',0:10:50,'font',fsize);set(gca,'yt',-6:2:8,'font',fsize);
nfft=2^nextpow2(l);y=fft(y,nfft)/l;f=fs/2*linspace(0,1,nfft/2+1);
subplot(2,1,2);plot(f,2*abs(y(1:nfft/2+1)));fize=15;axis([0 500 0 1]);
set(gca,'xt',0:100:500,'font',fsize);set(gca,'yt',0:.5:1,'font',fsize);
t(50) out of bounds (1), prints empty graph.
%}
%{
%a chaotic attractor of the power spectrum
a=1;b=.3;N=100000;x=zeros(1,N);y=zeros(1,N);x(1)=.1;y(1)=.1;
for n=1:N;x(n+1)=1-a*(y(n))^2+b*x(n);y(n+1)=x(n);end
subplot(2,1,1);plot(x(10:N),y(10:N),'.','Markersize',1);fsize=15;axis([-1 2 -1 2]);
set(gca,'xt',-1:1:2,'font',fsize);set(gca,'yt',-1:1:2,'font',fsize);
f=-N/2+1;N/2;freq=f*2/N;pow=abs(fft(x,N).^2);
subplot(2,1,2);plot(freq,log(pow));pma=20;axis([0 1 -10 pmax]);
set(gca,'xt',0:.5:1,'font',fsize);set(gca,'yt',-10:5:20,'font',fsize);
%}
%{
files=glob('destination of the file');
for i=1:numel(files);[~, name] = fileparts (files{i});
eval(sprintf('%s = load("%s", "-ascii");', name, files{i}));end
input_pyramid=double(rgb2gray(imread('pyramid.jpg')));
fft_pyramid=fft2(input_pyramid);
u=0:(size(fft_pyramid,1)-1);v=0:(size(fft_pyramid,2)-1);
idx=find(u>size(fft+pyramid,1)/2);u(idx)=u(idx)-size(fft_pyramid,1);
idy=find(v>size(fft_pyramid,2)/2);v(idy)=v(idy)-size(fft_pyramid,2);
[V,U]=meshgrid(v,u);D=sqrt(U.^2+V.^2);filter=double(D<=50);
fft_pyramid_blur=fft_pyramid.*filter;pyramid_blurred=ifft2(fft_pyramid_blurr);
pyramid_baa=[input_pyramid pyramid_blurred];pyramid_baa=255*mat2gray(pyramid_baa);
fft_pyramid_abs=log(abs(fftshift(fft_pyramid)));
fft_pyramid_blurr_abs=log(abs(fftshift(fft_pyramid_blurr)));
pyramid_ffts=[fft_pyramid_abs fft_pyramid_blurr_abs];
pyramid_ffts(pyramid_ffts==-Inf)=0;pyramid_ffts=255*mat2gray(pyramid_ffts);
pyramid_combined=[pyramid_baa;pyramid_ffts];figure;
imagesc(pyramid_combined);axis image; colormap gray; axis off;
%Unable to find 'pyramd'.jpg
%}
%End of Section 7
%{
clear;figure;
deqn1=@(t,P) P(1)*(100-P(1))/1000;
[t,P1]=ode45(deqn1,[0 100],50);[t1,P2]=ode45(deqn1,[0 100],150);
hold on;
plot(t,P1(:,1));plot(t1,P2(:,1));
axis([0 100 0 200]);fsize=15;
set(gca,'xtick',0:20:100,'Fontsize',fsize);
set(gca,'ytick',0:50:200,'fontsize',fsize);
hold off; % an initial value graph 
%}
%{
k=.00713;
deqn=@(t,c) k^(4-c)^2*(1-c/2);[t,ca]=ode45(deqn,[0 400],0);
plot(t,ca(:,1));axis([0 400 0 3]);fsize=15;
set(gca,'xtick',0:100:400,'fontsize',fsize);
set(gca,'ytick',0:1:3,'fontsize',fsize);hold off; % ?
%}
%{
hold on;
deqn=@(t,x) [x(2);-x(1)-2*(x(1)^2-1)*x(2)];
[t,xa]=ode45(deqn,[0 50],[5 0]);fsize=15;
plot(t,xa(:,1),'r');axis([0 .08 4.994 5]);
set(gca,'xtick',0:.02:.08,'fontsize',fsize);
set(gca,'ytick',4.994:.001:5,'fontsize',fsize);
explot('5-5*t^2/2+40*t^3-11515*t^4/24+9183*t^5/2',[0 .08]);
hold off; % graph of a van der pol equation.
%} 
%End of Section 8
%syms t x;
%{
clear;figure;
deq1=@(t,x) [2*x(1)+x(2);x(1)+2*x(2)];
vectorfield(deq1,-3:.25:3,-3:.25:3);hold on;
for x0=-3:1.5:3;for y0=-3:1.5:3;
[ts,xs]=ode45(deq1,[0 5],[x0 y0]);plot(xs(:,1),xs(:,2));end end
for x0=-3:1.5:3;for y0=-3:1.5:3;
[ts,xs]=ode45(deq1,[0 -5],[x0 y0]);plot(xs(:,1),xs(:,2));end end
axis([-3 3 -3 3]);fsize=15;
set(gca,'xtick',-3:1:3,'fontsize',fsize);
set(gca,'ytick',-3:1:3,'fontsize',fsize);hold off;
requires 'vectorfield.m', but file location not provided.
%}
%{
http://www2.math.umd.edu/~petersd/241/html/ex27.html#1
files must be in same folder as this file.
make sure names r exact same.
all files copied/saved give errors. (!?!) .m shuldnt need other .m anyway.
ignore district 9, move to district 10.
also, for sym, for sys use deq.
%} 
%End of Sections 9
%{
clear;figure;
deq1=@(t,x)[1-x(2);1-x(1)^2];
xmin=-3;xmax=4;sep=.9;
vectorfield(deq1,xmin:.5:xmax,xmin:.5:xmax);hold on;
for x0=xmin:sep:xmax;for y0=xmin:sep:xmax;
[ts,xs]=ode45(deq1,[0 10],[x0 y0]);plot(xs(:,1),xs(:,2));end end
for x0=xmin:sep:xmax;for y0=xmin:sep:xmax;
[ts,xs]=ode45(deq1,[0 -10],[x0 y0]);plot(xs(:,1),xs(:,2));end end
hold off;axis([xmin xmax xmin xmax]);fsize=15;title('Ham');
set(gca,'xtic',xmin:sep:xmax,'fontsize',fsize);
set(gca,'ytik',xmin:sep:xmax,'fontsize',fsize);hold off;
%}
%{
clear;hold on;
deq2=@(t,x) [x(1)*(1-x(1)/7)-6*x(1)*x(2)/(7+7*x(1));.2*x(2)*(1-.5*x(2)/x(1))];
options=odeset('reltol',1e-4,'abstol',1e-4);
[t,xa]=ode45(deq2,[0 100],[7.1 .1],options);
plot(xa(:,1),xa(:,2));axis([0 8 0 5]);fsize=15;
set(gca,'xtix',0:2:8,'font',fsize);set(gca,'ytix',0:2.5:5,'font',fsize);
hold off; % the time series holling tanner model.
%}
%{
clear;hold on; % holling tanner interacting species
deq2=@(t,x) [x(1)*(1-x(1)/7)-6*x(1)*x(2)/(7+7*x(1));.2*x(2)*(1-.5*x(2)/x(1))];
options=odeset('reltol',1e-4,'abstol',1e-4);
[t,xa]=ode45(deq2,[0 200],[7.1 .1],options);
plot(t,xa(:,1),'r');plot(t,xa(:,2),'b');
legend('prey','predator');axis([0 200 0 8]);fsize=15;
set(gca,'xt',0:50:200,'font',fsize);set(gca,'yt','font',fsize);hold off;
%}
%End of Section 10
%{
clear;figure;hold on;
ezplot('25*2^(-1/(0.015*x+0.00001)^2)+0.05*x+5*(320/y)^3-0.282*x',[10 100 200 500])
ezplot('25*2^(-1/(0.015*x+0.00001)^2)+0.05*x+5*(320/y)^3-0.05*y',[10 100 200 500])
alpha = 3; delta = 0.05; s = 0.282;
the kaldor business cycle model
deq2=@(t,x)[alpha*(25*2^(-1/(.015*x(1)+.00001)^2)+.05*x(1)+5*(320/x(2))^3-s*x(1));
25*2^(-1/(.015*x(1)+.00001)^2)+.05*x(1)+5*(320/x(2))^3-delta*x(2)];
the stable limit cycle.
[~,xs] = ode23s(sys,[0 100],[25 300]);plot(xs(:,1),xs(:,2),'r')
 Plot the unstable limit cycle.
[t,xb] = ode23s(sys,[0 -100],[60 355]);plot(xb(:,1),xb(:,2),'b');hold off;
axis([0 100 200 500]);fsize=15;
set(gca,'xt',0:20:100,'Font',fsize);set(gca,'yt',200:50:400,'Font',fsize);
%}
%{
deq1=@(t,x)[x=cos(t)^3;x(0)=0];%perturbation limit cycles
epsilon=.01;deq2=@(t,x)[x(2);-x(1)+epsilon*x(1)^3];
[t,xa]=ode45(deq2,[0,100],[1,0]);
subplot(2,1,1);plot(t,xa(:,1)-cos(t));
ylim=.5;axis([0 100 -ylim ylim]);fsize=15;
set(gca,'xt',0:10:100,'font',fsize);set(gca,'yt',-ylim:.2:ylim,'font',fsize);
subplot(2,1,2);plot(t,xa(:,1)-cos(t)-epsilon*(cos(t)/8-cos(t).^3/8+(3*t.*sin(t))/8));
ylim=.18;axis([0 100 -ylim ylim]);fsize=15;
set(gca,'xt',0:10:100,'font',fsize);set(gca,'yt',-ylim:.09:ylim,'font',fsize);
%}
%End of section 11
%{
figure;fsize=15;ezsurfc('-x^2/2+x^4/4+y^2/2',[-1.5,1.5]);
set(gca,'xt',-1.5:.5:1.5,'font',fsize);set(gca,'yt',-1.5:.5:1.5,'font',fsize);
figure;ezcontour('-x^2/2+x^4/4+y^2/2',[-1.5,1.5]); % a lyapunov hamiltonian
%}
%{
figure;fsize=15;% a lyapunov 'hopfield' network
ezsurf('-(x^2+y^2)-4*(log(cos(pi*x/2))+log(cos(pi*y/2)))/(.7*pi^2)',[-1,1,-1,1]);
axis([-1 1 -1 1 -.5 1]);
set(gca,'xt',-1:.5:1,'font',fsize);set(gca,'yt',-1:.5:1,'font',fsize);
figure;
x=-1:.01:1;y=-1:.01:1;[X,Y]=meshgrid(x,y);
Z=-(X.^2+Y.^2)-4*(log(cos(pi*X/2))+log(cos(pi*Y/2)))./(.7*pi^2);
contour(X,Y,Z,-1:.01:1);
set(gca,'xt',-1:.5:1,'font',fsize);set(gca,'yt',-1:.5:1,'font',fsize);
%}
%End of Section 12
%{
clear;axis tight;
x=-4:.1:4;
for n=1:9;plot(x,(n-5)-x.^2,x,0);M(n)=getframe;end
movieview(M);% animation of -x^2
%}
%{
r=0:.01:2;mu=.28*r.^6-6.^4+r.^2;
deq1=@(x,y)[mu*x-x^3','-y'];
plot(mu,r);fsize=15;% finding critical points exercise
%}
%{
function sys=prog_set13(~,x);
syms x y;x=0;y=0;
global mu;
X=x(1,:);Y=x(2,:);
P=Y+mu*X-X.*Y.^2;Q=mu*Y-X-Y.^3;sys=[P;Q];
%}
%{
clear;global mu;
for j=1:48;mu=j/40-1;
options=odeset('RelTol',1e-4,'AbsTol',1e-4);
x0=.5;y0=.5;[t,x]=ode45(@Prog_set13,[0 80],[x0 y0],options);
plot(x(:,1),x(:,2),'b');
axis([-1 1 -1 1]);fsize=15;
set(gca,'xt',-1:.2:1,'font',fsize);set(gca,'yt',-1:.2:1,'font',fsize);
F(j)=getframe;end
movie(F,5);
%} 
%End of Section 13
%{
sigma=10;r=28;b=8/2;
Lorenz=@(t,x)[sigma*(x(2)-x(1));r*x(1)-x(2)-x(1)*x(3);x(1)*x(2)-b*x(3)];
options=odeset('Rel',1e-4,'Abs',1e-4);
[t,xa]=ode45(Lorenz,[0 100],[15,20,30],options);
plot3(xa(:,1),xa(:,2),xa(:,3));fsize=15;%the lorenz attractor
%}
%{
chua=@(t,x)[15*(x(2)-x(1)-(-(5/7)*x(1)+(1/2)*(-(8/7)-(-5/7))*(abs(x(1)+1)-abs(x(1)-1))));x(1)-x(2)+x(3);-25.58*x(2)];
options=odeset('rel',1e-4,'abs',1e-4);[t,xb]=ode45(chua,[0 100],[-1.6,0,1.6],options);
plot3(xb(:,1),xb(:,2),xb(:,3));%the chua double scroll attractor
%}
%the chapman cycle, not attempting
%lyupanov exponents, not attempting
%{
fsize=15;A=.06;B=.02;f=1;k1=1.28;k2=2.4e6;k3=33.6;k4=3e3;kc=1;
epsilon=(kc*B)/(k3*A);epsilondash=(2*kc*k4*B)/(k2*k3*A);
q=(2*k1*k4*B)/(k2*k3*A);
BZReaction=@(t,x)[(q*x(2)-x(1)*x(2)+x(1)*(1-x(1)))/epsilon;
(-q*x(2)-x(1)*x(2)+f*x(3))/epsilondash;x(1)-x(3)];
options=odeset('Rel',1e-6,'Abs',1e-6);
[t,xa]=ode45(BZReaction,[0 50],[0,0,.1],options);
subplot(3,1,1);plot(t,xa(:,1));
subplot(3,1,2);plot(t,xa(:,2),'r');
subplot(3,1,3);plot(t,xa(:,3),'m');
%}
%{
clear;clc;figure;%a chua bifurication system
m1=-1/7;m2=2/7;
axis([-3 3 -.5 .5 -4 4]);fsize=15;view([20,50]);
for j=1:500;F(j)=getframe;a=(8+j*.006);
chua=@(t,x)[a*(x(2)-(m1*x(1)+(m1-m2)/2*(abs(x(1)+1)-abs(x(1)-1))));
x(1)-x(2)+x(3);-15*x(2)];
options=odeset('Rel',1e-4,'Abs',1e-4);
[t,xb]=ode45(chua,[0 100],[1.96,-.0519,-3.077],options);
plot3(xb(:,1),xb(:,2),xb(:,3));
F(j)=getframe;end
movie(F,5);%data values greater than float capacity
%}
%End of Section 14
%{
syms t r Dr;
Dr=-r^2;r(0)=1;
deq=inline('[-(r(1))^2]','t','r');
options=odeset('RelTol',1e-6,'AbsTol',1e-6);
[t,returns]=ode45(deq,0:2*pi:16*pi,1);
returns;% it does nothing
%}
%{
%detection of quasiperiodic behavior in poincaire hamiltonians
deq=@(t,p)[-sqrt(2)*p(3);-p(4);sqrt(2)*p(1);p(2)];
options=odeset('RelTol',1e-4,'AbsTol',1e-4);
[~,pp]=ode45(deq,[0 200],[.5,1.5,.5,0],options);
%a 3d projection:
subplot(2,1,1);fsize=15;plot3(pp(:,1),pp(:,2),pp(:,4));
deq=@(t,p)[-sqrt(2)*p(3);-p(4);sqrt(2)*p(1);p(2)];
options=odeset('RelTol',1e-2,'AbsTol',1e-4);
[t,pq]=ode45(deq,[0 600],[.5,1.5,.5,0],options);
%a 2d projection, determines where when q2=0,f1=f2:
k=0;p1_0=zeros(1,10^6);q1_0=zeros(1,10^6);
for i=1:size(pq);if abs(pq(i,4))<.1;
k=k+1;p1_0(k)=pq(i,1);q1_0(k)=pq(i,3);end end
subplot(2,1,2);hold on;axis([-1 1 -1 1]);
plot(p1_0(1:k),q1_0(1:k),'+','MarkerSize',3);hold off;
%}
%{
%a nontonomous poincare system
deq=@(t,x)[x(2);x(1)-.3*x(2)-(x(1))^3+.5*cos(1.25*t)];
options=odeset('RelTol',1e-4,'AbsTol',1e-4);
[t,xx]=ode45(deq,[0 200],[1,0],options);
plot(xx(:,1),xx(:,2));fsize=15;axis([-2 2 -2 2]);
%}
%{
clear;Gamma=.5;
deq=@(t,x)[x(2);x(1)-.3*x(2)-(x(1))^3+Gamma*cos(1.25*t)];
options=odeset('RelTol',1e-4,'AbsTol',1e-4);
[t,xx]=ode45(deq,0:(2/1.25)*pi:(6000/1.25)*pi,[1,0]);
plot(xx(:,1),xx(:,2),'.','markersize',1);
fsize=15;axis([-2 2 -2 2]);%a poincare duffing system graph(slow)
%}
%{
%supposed to be a phase portrait
deq=@(t,p)[-2*p(3);-2*p(4);2*p(1);2*p(2)];
options=odeset('RelTol',1e-4,'AbsTol',1e-4);
[t,pp]=ode45(deq,[0 100],[.5,1.5,.5,0],options);
subplot(2,1,1);fsize=15;plot3(pp(:,1),pp(:,2),pp(:,4));
k=0;p1_0=zeros(1,10^6);q1_0=zeros(1,10^6);
for n=1:1000;if abs(pp(n,4))<1000;
k=k+1;p1_0(k)=pp(n,1);q1_0(k)=pp(n,3);end end
subplot(2,1,2);hold on;axis([-1 1 -1 1]);
plot(p1_0(1:k),q1_0(1:k),'+','markersize',3);hold off;
%}
%15f programs requires functions, not attempting. End of Section 15
%{
clear;figure;hold on;
plot(0,0,'b.','markersize',15);
a1=.01;a2=1;b2=1;a3=1/3;xmin=-.4;ymin=-.4;ymax=.5;
options=odeset('reltol',1e-8,'abstol',1e-8);
sys=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2];
[t,xs]=ode45(sys,[0 50],[.24 0],options);plot(xs(:,1),xs(:,2),'r');
sys=@(t,x)[x(2)-a1*x(1)-a2*x(2)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2];
[t,xs]=ode45(sys,[0 50],[.2465 0],options);plot(xs(:,1),xs(:,2),'b','width',2);
sys=@(t,x)[x(2)-a1*x(1)-a2*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2];
[t,xs]=ode45(sys,[0 50],[.25 0],options);plot(xs(:,1),xs(:,2),'r');
axis([xmin xmax ymin ymax]);fsize=15;hold off;%an egg shaped limit cycle
%}
%{ 
clear;figure;hold on;
plot(0,0,'b.','markersize',15);
a1=.01;a2=1;b2=1;a3=1/3;b3=2;xmin=-1;xmax=1;ymin=-.6;ymax=1.3;
options=odeset('reltol',1e-8,'abstol',1e-8);
sys1=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys1,[0 100],[.28 0],options);plot(xs(:,1),xs(:,2),'r');
sys2=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys2,[0 100],[.295 0],options);plot(xs(:,1),xs(:,2),'b','linewidth',4);
sys3=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys3,[0 100],[.3 0],options);plot(xs(:,1),xs(:,2),'r');
sys4=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys4,[0 100],[.48 0],options);plot(xs(:,1),xs(:,2),'r');
sys5=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys5,[0 50],[.519 0],options);plot(xs(:,1),xs(:,2),'b','linewidth',4);
sys6=@(t,x)[x(2)-a1*x(1)-a2*x(1)^2-a3*x(1)^3;-x(1)-a1*x(2)-b2*x(1)^2-b3*x(1)^3];
[t,xs]=ode45(sys,[0 50],[.54 0],options);plot(xs(:,1),xs(:,2),'r');
axis([xmin xmax ymin ymax]);fsize=15;hold off;%egg shaped graph of gradients 
%}
%entire section 17 is missing (?)
%{
function prog_set18;
load housing.txt;
x=housing(:,[6 9 13]);t=housing(:,14);xmean=mean(x);xstd=std(x);
x=(x-ones(size(x,1),1)*xmean)./(ones(size(x,1),1)*xstd);x=[ones(size(x,1),1) x];
tmean=(max(t)+min(t))/2;tstd=(max(t)-min(t))/2;t=(t-tmean)/tstd;
%random weight vector
%rng('default');rng(123456);
w(:,1)=.1*randn(size(x,2),1);
y1=tanh(x*w(:,1));e1=t-y1;mse(1)=var(e1);
%do numepochs iterations
epochs=50;patterns=size(x,1);eta=.001;k=1;
for m=1:epochs;n=1:patterns;
yk=tanh(x(n,:)*w(:,k));err=yk-t(n);g=x(n,:)'*((1-yk^2)*err);
w(:,k+1)=w(:,k)-eta*g;k=k+1; end end
figure;
for m=1:size(w,1);plot(1:k,w(m,:));hold on;end
fsize=15;hold off;
%rng not yet implemented in octave 
%}
%{
clear;%the minimal chaotic attractor for a neuromodule
b1=-2;b2=3;w11=-20;w21=-6;w12=6;a=1;N=10000;x(1)=0;y(1)=2;
x=zeros(1,N);y=zeros(1,N);
for n=1:N;
x(n+1)=b1+w11*1/(1+exp(-a*x(n)))+w12*1/(1+exp(-a*y(n)));
y(n+1)=b2+w21*1/(1+exp(-a*x(n)));end 
hold on;
axis([-15 5 -5 5]);plot(x(50:N),y(50:N),'.','markersize',1);fsize=15;hold off;
%}
%{
clear all;format long;%just a graph of 2 bent lines.
N=2000;halfN=N/2;N1=1+halfN;max=10;a=1;al=.3;b2=3;w11=7;w12=-4;w21=5;
start=5;x=zeros(1,N);y=zeros(1,N);x(1)=-10;y(1)=-3;
for n=1:halfN;b1=(start+n*max/halfN);
x(n+1)=b1+w11*(exp(a*x(n))-exp(-a*x(n)))/(exp(a*x(n))+exp(-a*x(n)))+w12*(exp(al*y(n))-exp(-al*y(n)))/(exp(al*y(n))+exp(-al*y(n)));
y(n+1)=b2+w21*(exp(a*x(n))-exp(-a*x(n)))/(exp(a*x(n))+exp(-a*x(n)));end
for n=N1:N;b1=(start+2*max-n*max/halfN);
x(n+1)=b1+w11*(exp(a*x(n))-exp(-a*x(n)))/(exp(a*x(n))+exp(-a*x(n)))+w12*(exp(al*y(n))-exp(-al*y(n)))/(exp(al*y(n))+exp(-al*y(n)));
y(n+1)=b2+w21*(exp(a*x(n))-exp(-a*x(n)))/(exp(a*x(n))+exp(-a*x(n)));end 
fsize=14;subplot(2,1,1);hold on;plot(x(1:N),'-','markersize',1,'color','k');
hold off;x1=zeros(1,N);w=zeros(1,N);
for n=1:halfN;x1(n)=x(N+1-n);w(n)=start+n*max/halfN;end
subplot(2,1,2);hold on;
plot(w(1:halfN),x(1:halfN),'-','markersize',1,'color','k');
plot(2(1:halfN),x1(1:halfN),'-','markersize',1,'color','r');hold off;
%}
%End of section 18
%{
clear;
max=200;mu=4;k=.217;x=zeros(1,max);x(1)=.6;
for n=1:2:max;
x(n+1)=mu*x(n)*(1-x(n));x(n+2)=mu*x(n+1)*(1-x(n+1));
if n>60;
x(n+1)=k*mu*x(n)*(1-x(n));x(n+2)=mu*x(n+1)*(1-x(n+1));end end
hold on;plot(1:max,x(1:max));plot(1:max,x(1:max),'o');
fsize=15;hold off;%control of chaos in logistics
%}
%{
clear;%control of choas in a henon map
a=-1.2;b=.4;k1=-1.8;k2=1.2;xstar=.9358;ystar=xstar;
N=4;x=zeros(1,N);y=zeros(1,N);rsqr=zeros(1,N);
x(1)=.5;y(1)=.6;rsqr(1)=(x(1))^2+(y(1))^2;
for n=1:N;if n>198;
x(n+1)=(-k1*(x(n)-xstar)-k2*(y(n)-ystar)+a)+b*y(n)-(x(n))^2;
y(n+1)=x(n);else
x(n+1)=a+b*y(n)-(x(n))^2;y(n+1)=x(n);end
rsqr(n+1)=(x(n+1))^2+(y(n+1))^2;end
hold on;axis([0 N 0 6]);
plot(1:N,rsqr(1:N));plot(1:N,rsqr(1:N),'o');fsize=15;hold off;
%}
%{
clear;figure;%sync of 2 lorenz systems
sig=16;b=4;r=45.92;
lorenz=@(t,x)([sig*(x(2)-x(1));-x(1)*x(3)+r*x(1)-x(2);x(1)*x(2)-b*x(3);
-x(1)*x(5)+r*x(1)-x(4);x(1)*x(4)-b*x(5)]);
options=odeset('reltol',1e-6,'abstol',1e-6);
[t,xa]=ode45(lorenz,[0 100],[15 20 30 10 20],options);
plot(xa(:,2),xa(:,4));fsize=15;
%}
%End of Section 19
%{
clear;figure;
I=6.3;gna=120;gk=36;g1=.3;vna=50;vk=-77;v1=-54.402;c=1;y=zeros(1,4);
HHdeq=@(t,y)[(I-gna*y(2)^3*y(3)*(y(1)-vna)-gk*y(4)^4*(y(1)-vk)-g1*(y(1)-v1))/c;
.1*(y(1)+40)/(1-exp(-.1*(y(1)+40)))*(1-y(2))-4*exp(-.0556*(y(1)+65))*y(2);
.07*exp(-.05*(y(1)+65))*(1-y(3))-1/(1+exp(-.1*(y(1)+35)))*y(3);
.01*(y(1)+55)/(1-exp(-.1*(y(1)+55)))*(1-y(4))-.125*exp(-.0125*(y(1)+65))*y(4)];
[t,ya]=ode45(HHdeq,[0 100],[15,.01,.5,.4]);
figure(1);plot(t,ya(:,1));axis([0 100 -80 40]);fsize=15;
figure(2);subplot(3,1,1);plot(t,ya(:,2),'k');
subplot(3,1,2);plot(t,ya(:,3),'r');
subplot(3,1,3);plot(t,ya(:,4),'g');%Hodgkin Huxley oscillation profile
%}
%m=0;y=0;p=0;ans=0;ans=1./(1+exp(m*y+p));%err: m not defined.
%not attempting
%{
ans=@(t,x)
(xdot(1)=x(2));
xdot(2)=kappa-.6*x(2)-sin(x(1));
xdot=[xdot(1);xdot(2)];
%}
%not attempting
%not attempting
%{
clear;figure;%a memristor hysteresis beizer
eta=1;l=1;roff=70;ron=1;p=10;T=20;w0=.6;
deqn=@(t,w)(eta*(1-(2*w-1)^(2*p))*sin(2*pi*t/T))/(roff-(roff-ron)*w);
options=odeset('reltol',1e-8,'abstol',1e-8);
[t,wa]=ode45(deqn,[0 T],w0,options);
plot(sin(2*pi*t/T),sin(2*pi*t/T)./(roff-(roff-ron)*wa(:,1)));
axis([-1 1 -.5 .5]);fsize=15;
%}
%{
clear;clc;figure;bj=.6;
for n=1:200;
F(n)=getframe;kappa=.6+n*.001;
jos=@(t,x)[x(2);kappa-bj*x(2)-sin(x(1))];
options=odeset('reltol',1e-4,'abstol',1e-4);
[t,xb]=ode45(jos,[0 100],[0,3],options);
plot(xb(:,1),xb(:,2));axis([0 50 -1 3]);fsize=15;F(n)=getframe;end
movie(F,5);%no axes to capture
%} 
#End of Section 20
%{
%% Time domain parameters
fs = 20500;    % Sampling frequency
dt = 1/fs;     % Time resolution
T = 30;        % Signal duration
t = 0:dt:T-dt; % Total duration
N = length(t); % Number of time samples
%% Signal generation
f0_1 = 100; % fundamental frequency of first sinusoid
f0_2 = 200; % fundamental frequency of second sinusoid
x1 = sin(2*pi*ezsurf('y^2/2-x^2/2+x^4/4',[-2,2],50)*t); % first sinusoid
x2 = sin(2*pi*ezplot('x^3-4*x','x^2',[-3,3])*t); % second sinusoid
% We want 200 Hz signal to last for half of a time, therefore zero-out
% second half - use the logical indexing of time variable
x2(t>5)=0;
% Now add two signals
x = x1+x2;
% Calculate spectrum
X = abs(fft(x))/N;
% Prepare frequency axis
freqs = (0:N-1).*(fs/N);
% Time domain signal plot
subplot(211)
plot(t, x)
grid on
% Spectrum plot
subplot(212)
plot(freqs, X)
grid on
%% Playing back signal
% Normalize the audio:
x = 0.99*x/max(abs(x));
% For MATLAB R2014a use audioplayer
player = audioplayer(x, fs);
play(player)
% For older versions use wavplay
% wavplay(x, fs);
%Code from robert bristow johnson
%Azzi Abdelmalek on 14 Jun 2014
amp=10 
fs=20500  % sampling frequency
duration=10
freq1=100;freq2=0;freq3=0;
values=0:1/fs:duration;
a=amp*sin(2*pi*freq1*values)
b=amp*sin(2*pi*freq2*values)
c=amp*sin(2*pi*freq3*values)
sound(a);sound(b);sound(c);
%}