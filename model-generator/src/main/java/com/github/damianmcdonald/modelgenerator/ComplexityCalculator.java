package com.github.damianmcdonald.modelgenerator;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class ComplexityCalculator {

    @Value("${complejidad.valor.greenfield}")
    private int complexityGreenfield;

    @Value("${complejidad.valor.vpcs}")
    private int complexityVpcs;

    @Value("${complejidad.valor.subredes}")
    private int complexitySubnets;

    @Value("${complejidad.valor.conectividad}")
    private int complexityConnectivity;

    @Value("${complejidad.valor.peering.con.conectividad}")
    private int complexityPeeringWithConnectivity;

    @Value("${complejidad.valor.peering.sin.conectividad}")
    private int complexityPeeringWithoutConnectivity;

    @Value("${complejidad.valor.directoryservice}")
    private int complexityDirectoryService;

    @Value("${complejidad.valor.advsecurity}")
    private int complexityAdvSecurity;

    @Value("${complejidad.valor.advlogging}")
    private int complexityAdvLogging;

    @Value("${complejidad.valor.advmonitoring}")
    private int complexityAdvMonitoring;

    @Value("${complejidad.valor.advbackup}")
    private int complexityAdvBackup;

    @Value("${complejidad.valor.vms.min}")
    private int complexityVmsMin;

    @Value("${complejidad.valor.vms.medium}")
    private int complexityVmsMedium;

    @Value("${complejidad.valor.vms.max}")
    private int complexityVmsMax;

    @Value("${complejidad.valor.buckets}")
    private int complexityBuckets;

    @Value("${complejidad.valor.bbdd.min}")
    private int complexityDatabasesMin;

    @Value("${complejidad.valor.bbdd.max}")
    private int complexityDatabasesMax;

    @Value("${complejidad.valor.elb}")
    private int complexityElb;

    @Value("${complejidad.valor.autoscripts}")
    private int complexityAutoscripts;

    @Value("${complejidad.valor.otroservicios.weight}")
    private double complexityOtherWServicesWeight;

    @Value("${complejidad.valor.administered.weight}")
    private double complexityAdministeredWeight;

    @Value("${maximum.vms}")
    private int maximumVms;

    @Value("${maximum.otroservicios}")
    private int maximumOtherServices;

    public int calculateBaseComplexity(final OfferDto offerDto, final Phase phase) {
        int complexityCounter = 0;

        // greenfield
        if(offerDto.getGreenfield() == 0) {
            complexityCounter = complexityCounter + complexityGreenfield;
        }

        // vpcs
        if(offerDto.getVpc() == 1) {
            complexityCounter = complexityCounter + complexityVpcs;
        }

        // subnets
        if(offerDto.getSubnets() > 0.5) {
            complexityCounter = complexityCounter + complexitySubnets;
        }

        //connectivity
        if(offerDto.getConnectivity() == 1) {
            complexityCounter = complexityCounter + complexityConnectivity;
        }

        // peerings
        if(offerDto.getPeerings() == 1 && offerDto.getConnectivity() == 1) {
            complexityCounter = complexityCounter + complexityPeeringWithConnectivity;
        } else if (offerDto.getPeerings() == 1 && offerDto.getConnectivity() == 0) {
            complexityCounter = complexityCounter + complexityPeeringWithoutConnectivity;
        }

        // directory service
        if(offerDto.getDirectoryservice() == 1) {
            complexityCounter = complexityCounter + complexityDirectoryService;
        }

        // advanced security
        if(offerDto.getAdvsecurity() == 1) {
            complexityCounter = complexityCounter + complexityAdvSecurity;
        }

        // advanced logging
        if(offerDto.getAdvlogging() == 1) {
            complexityCounter = complexityCounter + complexityAdvLogging;
        }

        // advanced monitoring
        if(offerDto.getAdvmonitoring() == 1) {
            complexityCounter = complexityCounter + complexityAdvMonitoring;
        }

        // advanced backup
        if(offerDto.getAdvbackup() == 1) {
            complexityCounter = complexityCounter + complexityAdvBackup;
        }

        // vms
        if(offerDto.getVms() > 0 && offerDto.getVms() <= 0.3) {
            complexityCounter = complexityCounter + complexityVmsMin;
        } else if(offerDto.getVms() > 0.3 && offerDto.getVms() <= 0.6) {
            complexityCounter = complexityCounter + complexityVmsMedium;
        } else if(offerDto.getVms() > 0.6) {
            complexityCounter = complexityCounter + complexityVmsMax;
        }

        // buckets
        if(offerDto.getBuckets() > 0) {
            complexityCounter = complexityCounter + complexityBuckets;
        }

        // databases
        if(offerDto.getDatabases() > 0 && offerDto.getDatabases() <= 0.5) {
            complexityCounter = complexityCounter + complexityDatabasesMin;
        } else if(offerDto.getDatabases() > 0.5) {
            complexityCounter = complexityCounter + complexityDatabasesMax;
        }

        // elb
        if(offerDto.getElb() == 1) {
            complexityCounter = complexityCounter + complexityElb;
        }

        // auto scripts
        if(offerDto.getAutoscripts() == 1) {
            complexityCounter = complexityCounter + complexityAutoscripts;
        }

        // other services
        if(offerDto.getOtherservices() > 0) {
            final double otherServciesVal = Double.valueOf(offerDto.getOtherservices() * 5) * complexityOtherWServicesWeight;
            complexityCounter = complexityCounter + Double.valueOf(otherServciesVal).intValue();
        }

        // administered
        if(Phase.IMPLANTACION.equals(phase)) {
            if (offerDto.getAdministered() == 1 && offerDto.getVms() > 0) {
                final double scaledVms = offerDto.getVms() * 10.0;
                final double administeredScore = Math.ceil(scaledVms * complexityAdministeredWeight);
                final int administeredVms = new Double(administeredScore).intValue();
                final int administeredComplexity = administeredVms > 1 ? administeredVms : 1;
                complexityCounter = complexityCounter + administeredComplexity;
            }
        }

        return complexityCounter;
    }

    public int getMaxComplexity(final boolean isAdministered) {

        int maxComplexityCounter = 0;

        // greenfield
        maxComplexityCounter += complexityGreenfield;

        // vpcs
        maxComplexityCounter += complexityVpcs;

        // subnets
        maxComplexityCounter += complexitySubnets;

        //connectivity
        maxComplexityCounter += complexityConnectivity;

        // peerings
        maxComplexityCounter += complexityPeeringWithConnectivity;

        // directory service
        maxComplexityCounter += complexityDirectoryService;

        // advanced security
        maxComplexityCounter += complexityAdvSecurity;

        // advanced logging
        maxComplexityCounter += complexityAdvLogging;

        // advanced monitoring
        maxComplexityCounter += complexityAdvMonitoring;

        // advanced backup
        maxComplexityCounter += complexityAdvBackup;

        // vms
        maxComplexityCounter += complexityVmsMax;

        // buckets
        maxComplexityCounter += complexityBuckets;

        // databases
        maxComplexityCounter += complexityDatabasesMax;

        // elb
        maxComplexityCounter += complexityElb;

        // auto scripts
        maxComplexityCounter += complexityAutoscripts;

        // other services
        maxComplexityCounter += maximumOtherServices * Double.valueOf(complexityOtherWServicesWeight).intValue();

        // administered
        if(isAdministered) {
            maxComplexityCounter += Double.valueOf(Math.ceil(maximumVms * complexityAdministeredWeight)).intValue();
        }

        return maxComplexityCounter;

    }

}
