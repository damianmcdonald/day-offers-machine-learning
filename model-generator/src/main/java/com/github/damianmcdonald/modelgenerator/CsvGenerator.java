package com.github.damianmcdonald.modelgenerator;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Set;

@Service
public class CsvGenerator {

    @Value("${model.file.path}")
    private String modelFilePath;

    public final static String[] HEADERS = {
            "greenfield",
            "vpc",
            "subnets",
            "connectivity",
            "peerings",
            "directoryservice",
            "otherservices",
            "advsecurity",
            "advlogging",
            "advmonitoring",
            "advbackup",
            "vms",
            "buckets",
            "databases",
            "elb",
            "autoscripts",
            "administered",
            "phase1prediction",
            "phase2prediction",
            "phase3prediction",
            "phase4prediction"
    };

    public void createCSVFile(Set<Offer> offers) throws IOException {
        FileWriter out = new FileWriter(this.modelFilePath);
        try (CSVPrinter printer = new CSVPrinter(out, CSVFormat.DEFAULT
                .withHeader(HEADERS))) {
            offers.forEach((offer) -> {
                try {
                    printer.printRecord(
                            offer.getOfferDto().getGreenfield(),
                            offer.getOfferDto().getVpc(),
                            offer.getOfferDto().getSubnets(),
                            offer.getOfferDto().getConnectivity(),
                            offer.getOfferDto().getPeerings(),
                            offer.getOfferDto().getDirectoryservice(),
                            offer.getOfferDto().getOtherservices(),
                            offer.getOfferDto().getAdvsecurity(),
                            offer.getOfferDto().getAdvlogging(),
                            offer.getOfferDto().getAdvmonitoring(),
                            offer.getOfferDto().getAdvbackup(),
                            offer.getOfferDto().getVms(),
                            offer.getOfferDto().getBuckets(),
                            offer.getOfferDto().getDatabases(),
                            offer.getOfferDto().getElb(),
                            offer.getOfferDto().getAutoscripts(),
                            offer.getOfferDto().getAdministered(),
                            offer.getPhase1prediction(),
                            offer.getPhase2prediction(),
                            offer.getPhase3prediction(),
                            offer.getPhase4prediction()
                    );
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        }
    }
}
