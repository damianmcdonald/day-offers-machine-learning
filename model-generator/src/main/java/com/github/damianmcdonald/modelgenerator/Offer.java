package com.github.damianmcdonald.modelgenerator;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

public class Offer {

    private OfferDto offerDto;
    private double phase1prediction;
    private double phase2prediction;
    private double phase3prediction;
    private double phase4prediction;


    private Offer(final OfferDto offerDto,
                    final double phase1prediction,
                    final double phase2prediction,
                    final double phase3prediction,
                    final double phase4prediction) {
        this.offerDto = offerDto;
        this.phase1prediction = phase1prediction;
        this.phase2prediction = phase2prediction;
        this.phase3prediction = phase3prediction;
        this.phase4prediction = phase4prediction;
    }

    @Override
    public int hashCode() {
        // you pick a hard-coded, randomly chosen, non-zero, odd number
        // ideally different for each class
        return new HashCodeBuilder(17, 37).
                append(offerDto.getGreenfield()).
                append(offerDto.getVpc()).
                append(offerDto.getSubnets()).
                append(offerDto.getConnectivity()).
                append(offerDto.getPeerings()).
                append(offerDto.getDirectoryservice()).
                append(offerDto.getOtherservices()).
                append(offerDto.getAdvsecurity()).
                append(offerDto.getAdvlogging()).
                append(offerDto.getAdvmonitoring()).
                append(offerDto.getAdvbackup()).
                append(offerDto.getVms()).
                append(offerDto.getBuckets()).
                append(offerDto.getDatabases()).
                append(offerDto.getElb()).
                append(offerDto.getAutoscripts()).
                append(offerDto.getAdministered()).
                toHashCode();
    }

    @Override
    public boolean equals(Object obj) {
        boolean equals = false;
        if ( obj != null &&
                Offer.class.isAssignableFrom(obj.getClass()) ) {
            final Offer rhs = (Offer) obj;
            equals = (new EqualsBuilder().
            append(offerDto.getGreenfield(), rhs.getOfferDto().getGreenfield()).
            append(offerDto.getVpc(), rhs.getOfferDto().getVpc()).
            append(offerDto.getSubnets(), rhs.getOfferDto().getSubnets()).
            append(offerDto.getConnectivity(), rhs.getOfferDto().getConnectivity()).
            append(offerDto.getPeerings(), rhs.getOfferDto().getPeerings()).
            append(offerDto.getDirectoryservice(), rhs.getOfferDto().getDirectoryservice()).
            append(offerDto.getOtherservices(), rhs.getOfferDto().getOtherservices()).
            append(offerDto.getAdvsecurity(), rhs.getOfferDto().getAdvsecurity()).
            append(offerDto.getAdvlogging(), rhs.getOfferDto().getAdvlogging()).
            append(offerDto.getAdvmonitoring(), rhs.getOfferDto().getAdvmonitoring()).
            append(offerDto.getAdvbackup(), rhs.getOfferDto().getAdvbackup()).
            append(offerDto.getVms(), rhs.getOfferDto().getVms()).
            append(offerDto.getBuckets(), rhs.getOfferDto().getBuckets()).
            append(offerDto.getDatabases(), rhs.getOfferDto().getDatabases()).
            append(offerDto.getElb(), rhs.getOfferDto().getElb()).
            append(offerDto.getAutoscripts(), rhs.getOfferDto().getAutoscripts()).
            append(offerDto.getAdministered(), rhs.getOfferDto().getAdministered()))
            .isEquals();
        }
        return equals;
    }

    public OfferDto getOfferDto() {
        return offerDto;
    }

    public double getPhase1prediction() {
        return phase1prediction;
    }

    public double getPhase2prediction() {
        return phase2prediction;
    }

    public double getPhase3prediction() {
        return phase3prediction;
    }

    public double getPhase4prediction() {
        return phase4prediction;
    }

    public static class OfferBuilder {

        private OfferDto offerDto;
        private double phase1prediction;
        private double phase2prediction;
        private double phase3prediction;
        private double phase4prediction;

        public OfferBuilder offerDto(final OfferDto offerDto) {
            this.offerDto = offerDto;
            return this;
        }

        public OfferBuilder phase1prediction(final double phase1prediction) {
            this.phase1prediction = phase1prediction;
            return this;
        }

        public OfferBuilder phase2prediction(final double phase2prediction) {
            this.phase2prediction = phase2prediction;
            return this;
        }

        public OfferBuilder phase3prediction(final double phase3prediction) {
            this.phase3prediction = phase3prediction;
            return this;
        }

        public OfferBuilder phase4prediction(final double phase4prediction) {
            this.phase4prediction = phase4prediction;
            return this;
        }

        public Offer build() {
            return new Offer(offerDto,
                    phase1prediction,
                    phase2prediction,
                    phase3prediction,
                    phase4prediction);
        }

    }

}

