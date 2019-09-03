% Import IMUparams and raw data into matlab first

sx = IMUParams(1,1)
sy = IMUParams(1,2)
sz = IMUParams(1,3)
bx = IMUParams(2,1)
by = IMUParams(2,2)
bz = IMUParams(2,3)


ax = ((valt(:,1))*sx+bx)*9.81
ay = ((valt(:,2))*sy+by)*9.81
az = ((valt(:,3))*sz+bz)*9.81

phi = atan2(ay,sqrt(ax.^2+az.^2))
theta = atan2(ax,sqrt(ay.^2+az.^2))
psi = atan2(sqrt(ax.^2+ay.^2),az)

plot(phi)
hold on
plot(theta)
plot(psi)