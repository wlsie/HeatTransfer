import numpy as np
from CoolProp.HumidAirProp import HAPropsSI

def plainFinEfficiency(tubeArrangement, HeatTransferCoefficient, FinThickness, 
                        ThermalConductivity, VerticalSpacing, HorizontalSpacing, TubeOuterDiameter):
    alpha = HeatTransferCoefficient
    delta = FinThickness
    k = ThermalConductivity
    Pt = VerticalSpacing
    Pl = HorizontalSpacing
    Do = TubeOuterDiameter
    
    m = np.sqrt(2*alpha/(k*delta))
    XT = Pt/2
    XL = Pl/2
    XD = np.sqrt(XT**2+Pl**2)/2
    
    if(tubeArrangement.lower()=='inline'):
        Req = 1.28*XT*np.sqrt((XL/XT)-0.2)
    else:
        Req = 1.27*XT*np.sqrt((XD/XT)-0.3)
        
    r = Do/2
    phi = (Req/r-1)*(1+0.35*np.log(Req/r))
    eta = np.tanh(m*r*phi)/(m*r*phi)          
    
    return eta
    
    

class PlainFinHX:
    def __init__(self,tubeArrangement,NtubesPerRow,Nrow,Lt,Do,Fp,Pl,Pt,deltaf):
        self.NtubesPerRow = NtubesPerRow
        self.Nrow = Nrow
        self.Do = Do
        self.Fp = Fp
        self.Pl = Pl
        self.Pt = Pt
        self.deltaf = deltaf        
        self.Lt = Lt
        
        self.Dc = 2*deltaf + Do
        self.L1 = Lt
        self.L2 = Nrow*Pl
        self.L3 = NtubesPerRow*Pt
        self.FPM = 1/(Fp)
        self.a = 0.5*((Pt-Do)-(Pt-Do)*deltaf*self.FPM)
        self.b = np.sqrt((Pt/2)**2+Pl**2)-Do-(Pt-Do)*deltaf*self.FPM
        self.c = 2*min(self.a,self.b)
        
        if(tubeArrangement.lower()=='inline'):
            self.Ao = NtubesPerRow*((Pt-Do)*self.L1-(Pt-Do)*deltaf*self.FPM*self.L1)
        else:
            self.Ao = self.L1*((NtubesPerRow-1)*self.c+(Pt-Do)-(Pt-Do)*deltaf*self.FPM)
            
        self.Ap = (np.pi*Do*(self.L1-deltaf*self.FPM*self.L1)*Nrow
                   *NtubesPerRow+2*(self.L2*self.L3-np.pi*Do*Do/4*Nrow*NtubesPerRow))
        self.As = (2*(self.L2*self.L3-np.pi/4*Do**2*Nrow*NtubesPerRow)
                   *self.FPM*self.L1+2*self.L3*deltaf*self.FPM*self.L1)
        self.Afr = self.L1*self.L3
        self.sigma = self.Ao/self.Afr
        self.At = (self.Ap+self.As)
        self.Dh = 4*self.Ao*self.L2/self.At


class AirThermoPhysicalProps:
    def __init__(self,PressurePa,TemperatureK,RHRatio):
        self.Temp = TemperatureK
        self.Rh = RHRatio
        self.P_amb = PressurePa
        self.rho = 1/HAPropsSI("Vha", "T", self.Temp, "P", self.P_amb, "R", self.Rh)
        self.mu = HAPropsSI("mu", "T", self.Temp, "P", self.P_amb, "R", self.Rh)
        self.cp = HAPropsSI("Cha", "T", self.Temp, "P", self.P_amb, "R", self.Rh)
        self.k = HAPropsSI("k", "T", self.Temp, "P", self.P_amb, "R", self.Rh)
        self.Pr = self.mu*self.cp/self.k



        
def CCWangPlainFinCalc(HX,AirProps,InletVelocity):
    u_max = InletVelocity/HX.sigma
    Re_Dc = AirProps.rho*u_max*HX.Dc/AirProps.mu
    P1 = 1.9-0.23*np.log(Re_Dc)
    P2 = -0.236+0.126*np.log(Re_Dc)
    P3 = -0.361-(0.042*HX.Nrow/np.log(Re_Dc))+0.158*np.log(HX.Nrow*np.power((HX.Fp/HX.Dc),0.41))
    P4 = -1.224 - (0.076*np.power((HX.Pl/HX.Dh),1.42))/np.log(Re_Dc)
    P5 = -0.083 + (0.058*HX.Nrow)/np.log(Re_Dc)
    P6 = -5.735+1.21*np.log(Re_Dc/HX.Nrow)
    if(HX.Nrow==1):
        j =(0.108*np.power(Re_Dc,-0.29)*np.power((HX.Pt/HX.Pl),P1)
            *np.power((HX.Fp/HX.Dc),-1.084)*np.power((HX.Fp/HX.Dh),-0.786)*np.power((HX.Fp/HX.Pt),P2))
    else:
        j=(0.086*np.power(Re_Dc,P3)*np.power(HX.Nrow,P4)
           *np.power((HX.Fp/HX.Dc),P5)*np.power((HX.Fp/HX.Dh),P6)*np.power((HX.Fp/HX.Pt),-0.93))
    G = AirProps.rho*u_max
    ho = j*G*AirProps.cp/(np.power(AirProps.Pr,(2/3)))
    F1 = -0.764+0.739*(HX.Pt/HX.Pl)+0.177*(HX.Fp/HX.Dc)-0.00759/HX.Nrow
    F2 = -15.689+64.021/np.log(Re_Dc)
    F3 = 1.696 - 15.695/np.log(Re_Dc)
    f = 0.0267*np.power(Re_Dc,F1)*np.power((HX.Pt/HX.Pl),F2)*np.power((HX.Fp/HX.Dc),F3)
    dP = f*G**2/(2*AirProps.rho)*(HX.As+HX.Ap)/HX.Ao
    return ho,dP


class WavyHerringboneFinHX:
    def __init__(self,tubeArrangement,NtubesPerRow,Nrow,Lt,Do,Fp,Pl,Pt,deltaf,Pd,Xf):
        self.NtubesPerRow = NtubesPerRow
        self.Nrow = Nrow
        self.Do = Do
        self.Fp = Fp
        self.Pl = Pl
        self.Pt = Pt
        self.deltaf = deltaf        
        self.Lt = Lt
        self.Pd = Pd
        self.Xf = Xf
        
        self.Dc = 2*deltaf + Do
        self.L1 = Lt
        self.L2 = Nrow*Pl
        self.L3 = NtubesPerRow*Pt
        self.FPM = 1/(Fp)
        self.Fs = Fp-deltaf
        self.a = 0.5*((Pt-Do)-(Pt-Do)*deltaf*self.FPM)
        self.b = np.sqrt((Pt/2)**2+Pl**2)-Do-(Pt-Do)*deltaf*self.FPM
        self.c = 2*min(self.a,self.b)
        self.theta_rad = np.arccos(self.Xf/(np.sqrt(self.Xf**2+self.Pd**2)))
        self.alpha = 1/np.cos(self.theta_rad)
        
        if(tubeArrangement.lower()=='inline'):
            self.Ao = NtubesPerRow*((Pt-Do)*self.L1-(Pt-Do)*deltaf*self.FPM*self.L1)
        else:
            self.Ao = self.L1*((NtubesPerRow-1)*self.c+(Pt-Do)-(Pt-Do)*deltaf*self.FPM)
            
        self.Ap = (np.pi*Do*(self.L1-deltaf*self.FPM*self.L1)*Nrow
                   *NtubesPerRow+2*(self.L2*self.L3-np.pi*Do*Do/4*Nrow*NtubesPerRow))
        self.As = (2*(self.L2*self.L3*self.alpha-np.pi/4*Do**2*Nrow*NtubesPerRow)
                   *self.FPM*self.L1+2*self.L3*self.alpha*deltaf*self.FPM*self.L1)
        self.Afr = self.L1*self.L3
        self.sigma = self.Ao/self.Afr
        self.At = (self.Ap+self.As)
        self.Dh = 4*self.Ao*self.L2/self.At

        
def CCWangWavyHerringboneFinCalc(HX,AirProps,InletVelocity):
    u_max = InletVelocity/HX.sigma
    Re_Dc = AirProps.rho*u_max*HX.Dc/AirProps.mu 

    if(Re_Dc<1000):
        J1 = (0.0045-0.491
              *(np.power(Re_Dc,-0.0316-0.0171*np.log(HX.Nrow*np.tan(HX.theta_rad))))
              *(np.power(HX.Pl/HX.Pt,-0.109*np.log(HX.Nrow*np.tan(HX.theta_rad))))
              *(np.power(HX.Dc/HX.Dh,0.542+0.0471*HX.Nrow))
              *(np.power(HX.Fs/HX.Dc,0.984))
              *np.power(HX.Fs/HX.Pt,-0.349))
        J2 = -2.72+6.84*np.tan(HX.theta_rad)
        J3 = 2.66*np.tan(HX.theta_rad)
        j = (0.882*np.power(Re_Dc,J1)*np.power(HX.Dc/HX.Dh,J2)*np.power(HX.Fs/HX.Pt,J3)
             *np.power(HX.Fs/HX.Dc,-1.58)*np.power(np.tan(HX.theta_rad),-0.2))
        F1 = (-0.574-0.137*np.power(np.log(Re_Dc)-5.26,0.245)
              *np.power(HX.Pt/HX.Dc,-0.765)
              *np.power(HX.Dc/HX.Dh,-0.243)
              *np.power(HX.Fs/HX.Dh,-0.474)
              *np.power(np.tan(HX.theta_rad),-0.217)
              *np.power(HX.Nrow,0.035))
        F2 = -3.05*np.tan(HX.theta_rad)
        F3 = -0.192*HX.Nrow
        F4 = -0.646*np.tan(HX.theta_rad)
        f = (4.37*np.power(Re_Dc,F1)*np.power(HX.Fs/HX.Dh,F2)
             *np.power(HX.Pl/HX.Pt,F3)*np.power(HX.Dc/HX.Dh,0.2054)*np.power(HX.Nrow,F4))        
    else:
        j1 = (-0.0545-0.0538*np.tan(HX.theta_rad)-0.302*np.power(HX.Nrow,-0.24)*np.power(HX.Fs/HX.Pl,-1.3)
              *np.power(HX.Pl/HX.Pt,0.379)*np.power(HX.Pl/HX.Dh,-1.35)*np.power(np.tan(HX.theta_rad),-0.256))
        j2 = (-1.29*np.power(HX.Pl/HX.Pt,1.77-9.43*np.tan(HX.theta_rad))
              *np.power(HX.Dc/HX.Dh,0.229-1.43*np.tan(HX.theta_rad))
              *np.power(HX.Nrow,-0.166-1.08*np.tan(HX.theta_rad))
              *np.power(HX.Fs/HX.Pt,-0.174*np.log(0.5*HX.Nrow)))
        j = (0.0646*np.power(Re_Dc,j1)*np.power(HX.Dc/HX.Dh,j2)*np.power(HX.Fs/HX.Pt,-1.03)
             *np.power(HX.Pl/HX.Dc,0.432)*np.power(np.tan(HX.theta_rad),-0.692)*np.power(HX.Nrow,-0.737)) 
        f1 = (-0.141*np.power(HX.Fs/HX.Pl,0.0512)*np.power(np.tan(HX.theta_rad),-0.472)
              *np.power(HX.Pl/HX.Pt,0.35)*np.power(HX.Pt/HX.Dh,0.449*np.tan(HX.theta_rad))
              *np.power(HX.Nrow,-0.049+0.237*np.tan(HX.theta_rad)))
        f2 = -0.562*np.power(np.log(Re_Dc),-0.0923)*np.power(HX.Nrow,0.013)
        f3 = 0.302*np.power(Re_Dc,0.03)*np.power(HX.Pt/HX.Dc,0.026)
        f4 = -0.306 + 3.63*np.tan(HX.theta_rad)
        f = (0.228*np.power(Re_Dc,f1)*np.power(np.tan(HX.theta_rad),f2)*np.power(HX.Fs/HX.Pl,f3)*np.power(HX.Pl/HX.Dc,f4)
             *np.power(HX.Dc/HX.Dh,0.383)*np.power(HX.Pl/HX.Pt,-0.247))    
        
    G = AirProps.rho*u_max
    ho = j*G*AirProps.cp/(np.power(AirProps.Pr,(2/3)))
    dP = f*G**2/(2*AirProps.rho)*(HX.As+HX.Ap)/HX.Ao
    return ho,dP

