package com.github.damianmcdonald.modelgenerator;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

public class OfferDto {

    private int greenfield;
    private double vpc;
    private double subnets;
    private int connectivity;
    private int peerings;
    private int directoryservice;
    private double otherservices;
    private int advsecurity;
    private int advlogging;
    private int advmonitoring;
    private int advbackup;
    private double vms;
    private double buckets;
    private double databases;
    private int elb;
    private int autoscripts;
    private int administered;

    private OfferDto(final int greenfield,
                     final double vpc,
                     final double subnets,
                     final int connectivity,
                     final int peerings,
                     final int directoryservice,
                     final double otherservices,
                     final int advsecurity,
                     final int advlogging,
                     final int advmonitoring,
                     final int advbackup,
                     final double vms,
                     final double buckets,
                     final double databases,
                     final int elb,
                     final int autoscripts,
                     final int administered) {
        this.greenfield = greenfield;
        this.vpc = vpc;
        this.subnets = subnets;
        this.connectivity = connectivity;
        this.peerings = peerings;
        this.directoryservice = directoryservice;
        this.otherservices = otherservices;
        this.advsecurity = advsecurity;
        this.advlogging = advlogging;
        this.advmonitoring = advmonitoring;
        this.advbackup = advbackup;
        this.vms = vms;
        this.buckets = buckets;
        this.databases = databases;
        this.elb = elb;
        this.autoscripts = autoscripts;
        this.administered = administered;
    }

    @Override
    public int hashCode() {
        // you pick a hard-coded, randomly chosen, non-zero, odd number
        // ideally different for each class
        return new HashCodeBuilder(55, 83).
                append(greenfield).
                append(vpc).
                append(subnets).
                append(connectivity).
                append(peerings).
                append(directoryservice).
                append(otherservices).
                append(advsecurity).
                append(advlogging).
                append(advmonitoring).
                append(advbackup).
                append(vms).
                append(buckets).
                append(databases).
                append(elb).
                append(autoscripts).
                append(administered).
                toHashCode();

    }

    @Override
    public boolean equals(Object obj) {
        boolean equals = false;
        if ( obj != null &&
                OfferDto.class.isAssignableFrom(obj.getClass()) ) {
            final OfferDto rhs = (OfferDto) obj;
            equals = (new EqualsBuilder().
            append(greenfield, rhs.greenfield).
            append(vpc, rhs.vpc).
            append(subnets, rhs.subnets).
            append(connectivity, rhs.connectivity).
            append(peerings, rhs.peerings).
            append(directoryservice, rhs.directoryservice).
            append(otherservices, rhs.otherservices).
            append(advsecurity, rhs.advsecurity).
            append(advlogging, rhs.advlogging).
            append(advmonitoring, rhs.advmonitoring).
            append(advbackup, rhs.advbackup).
            append(vms, rhs.vms).
            append(buckets, rhs.buckets).
            append(databases, rhs.databases).
            append(elb, rhs.elb).
            append(autoscripts, rhs.autoscripts).
            append(administered, rhs.administered))
            .isEquals();
        }
        return equals;
    }

    public int getGreenfield() {
        return greenfield;
    }

    public double getVpc() {
        return vpc;
    }

    public double getSubnets() {
        return subnets;
    }

    public int getConnectivity() {
        return connectivity;
    }

    public int getPeerings() {
        return peerings;
    }

    public int getDirectoryservice() {
        return directoryservice;
    }

    public double getOtherservices() {
        return otherservices;
    }

    public int getAdvsecurity() {
        return advsecurity;
    }

    public int getAdvlogging() {
        return advlogging;
    }

    public int getAdvmonitoring() {
        return advmonitoring;
    }

    public int getAdvbackup() {
        return advbackup;
    }

    public double getVms() {
        return vms;
    }

    public double getBuckets() {
        return buckets;
    }

    public double getDatabases() {
        return databases;
    }

    public int getElb() {
        return elb;
    }

    public int getAutoscripts() {
        return autoscripts;
    }

    public int getAdministered() {
        return administered;
    }

    public static class OfferDtoBuilder {

        private int greenfield;
        private double vpc;
        private double subnets;
        private int connectivity;
        private int peerings;
        private int directoryservice;
        private double otherservices;
        private int advsecurity;
        private int advlogging;
        private int advmonitoring;
        private int advbackup;
        private double vms;
        private double buckets;
        private double databases;
        private int elb;
        private int autoscripts;
        private int administered;

        public OfferDtoBuilder greenfield(final int greenfield) {
            this.greenfield = greenfield;
            return this;
        }

        public OfferDtoBuilder vpcs(final double vpc) {
            this.vpc = vpc;
            return this;
        }

        public OfferDtoBuilder subnets(final double subnets) {
            this.subnets = subnets;
            return this;
        }

        public OfferDtoBuilder connectivity(final int connectivity) {
            this.connectivity = connectivity;
            return this;
        }

        public OfferDtoBuilder peerings(final int peerings) {
            this.peerings = peerings;
            return this;
        }

        public OfferDtoBuilder directoryservice(final int directoryservice) {
            this.directoryservice = directoryservice;
            return this;
        }

        public OfferDtoBuilder otherservices(final double otherservices) {
            this.otherservices = otherservices;
            return this;
        }

        public OfferDtoBuilder advsecurity(final int advsecurity) {
            this.advsecurity = advsecurity;
            return this;
        }

        public OfferDtoBuilder advlogging(final int advlogging) {
            this.advlogging = advlogging;
            return this;
        }

        public OfferDtoBuilder advmonitoring(final int advmonitoring) {
            this.advmonitoring = advmonitoring;
            return this;
        }

        public OfferDtoBuilder advbackup(final int advbackup) {
            this.advbackup = advbackup;
            return this;
        }

        public OfferDtoBuilder vms(final double vms) {
            this.vms = vms;
            return this;
        }

        public OfferDtoBuilder buckets(final double buckets) {
            this.buckets = buckets;
            return this;
        }

        public OfferDtoBuilder databases(final double databases) {
            this.databases = databases;
            return this;
        }

        public OfferDtoBuilder elb(final int elb) {
            this.elb = elb;
            return this;
        }

        public OfferDtoBuilder autoscripts(final int autoscripts) {
            this.autoscripts = autoscripts;
            return this;
        }

        public OfferDtoBuilder administered(final int administered) {
            this.administered = administered;
            return this;
        }

        public OfferDto build() {
            return new OfferDto(greenfield,
                                vpc,
                                subnets,
                                connectivity,
                                peerings,
                                directoryservice,
                                otherservices,
                                advsecurity,
                                advlogging,
                                advmonitoring,
                                advbackup,
                                vms,
                                buckets,
                                databases,
                                elb,
                                autoscripts,
                                administered);
        }

    }

}
