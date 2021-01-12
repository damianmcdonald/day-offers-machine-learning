package com.github.damianmcdonald.modelgenerator;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

import java.io.IOException;
import java.util.Set;

@SpringBootApplication
public class ModelGeneratorApp {

    public static void main(String[] args) throws IOException {
        final ApplicationContext context = SpringApplication.run(ModelGeneratorApp.class, args);
        final OfferGenerator offerGenerator = context.getBean(OfferGenerator.class);
        final CsvGenerator csvGenerator = context.getBean(CsvGenerator.class);
        final Set<Offer> offers = offerGenerator.generateModel();
        csvGenerator.createCSVFile(offers);
    }

}
