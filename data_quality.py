from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SentimentDatasetQualityAssessment:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)
        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = self.dataset['test'].to_pandas()
        
        # Download NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def generate_quality_report(self):
        """
        Generate a comprehensive data quality assessment report
        """
        report = {
            'dataset_overview': self._dataset_overview(),
            'missing_values': self._check_missing_values(),
            'label_distribution': self._analyze_label_distribution(),
            'text_characteristics': self._analyze_text_characteristics(),
            'duplicate_analysis': self._check_duplicates(),
            'text_complexity': self._analyze_text_complexity()
        }
        
        return report

    def _dataset_overview(self):
        """
        Provide basic dataset overview
        """
        return {
            'dataset_name': self.dataset_name,
            'total_train_samples': len(self.train_df),
            'total_test_samples': len(self.test_df),
            'columns': list(self.train_df.columns)
        }

    def _check_missing_values(self):
        """
        Check for missing values
        """
        train_missing = self.train_df.isnull().sum()
        test_missing = self.test_df.isnull().sum()
        
        return {
            'train_missing': train_missing.to_dict(),
            'test_missing': test_missing.to_dict(),
            'total_missing_train': train_missing.sum(),
            'total_missing_test': test_missing.sum()
        }

    def _analyze_label_distribution(self):
        """
        Analyze label distribution
        """
        train_label_dist = self.train_df['label'].value_counts(normalize=True)
        test_label_dist = self.test_df['label'].value_counts(normalize=True)
        
        # Visualize label distribution
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        train_label_dist.plot(kind='bar')
        plt.title('Train Label Distribution')
        
        plt.subplot(1, 2, 2)
        test_label_dist.plot(kind='bar')
        plt.title('Test Label Distribution')
        plt.tight_layout()
        plt.savefig('label_distribution.png')
        plt.close()
        
        return {
            'train_label_distribution': train_label_dist.to_dict(),
            'test_label_distribution': test_label_dist.to_dict(),
            'label_balance_train': self._calculate_label_balance(train_label_dist),
            'label_balance_test': self._calculate_label_balance(test_label_dist)
        }

    def _calculate_label_balance(self, distribution):
        """
        Calculate label balance metric
        """
        return 1 - abs(distribution.max() - distribution.min())

    def _analyze_text_characteristics(self):
        """
        Analyze text characteristics
        """
        def text_stats(texts):
            lengths = texts.str.len()
            word_counts = texts.str.split().str.len()
            
            return {
                'avg_length': lengths.mean(),
                'median_length': lengths.median(),
                'max_length': lengths.max(),
                'min_length': lengths.min(),
                'avg_word_count': word_counts.mean(),
                'median_word_count': word_counts.median()
            }
        
        # Visualize text length distribution
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        self.train_df['text'].str.len().hist(bins=50)
        plt.title('Train Text Length Distribution')
        
        plt.subplot(1, 2, 2)
        self.test_df['text'].str.len().hist(bins=50)
        plt.title('Test Text Length Distribution')
        plt.tight_layout()
        plt.savefig('text_length_distribution.png')
        plt.close()
        
        return {
            'train_text_stats': text_stats(self.train_df['text']),
            'test_text_stats': text_stats(self.test_df['text'])
        }

    def _check_duplicates(self):
        """
        Check for duplicate entries
        """
        train_duplicates = self.train_df.duplicated()
        test_duplicates = self.test_df.duplicated()
        
        return {
            'train_total_duplicates': train_duplicates.sum(),
            'test_total_duplicates': test_duplicates.sum(),
            'train_duplicate_percentage': train_duplicates.mean() * 100,
            'test_duplicate_percentage': test_duplicates.mean() * 100
        }

    def _analyze_text_complexity(self):
        """
        Analyze text complexity
        """
        def text_complexity(texts):
            stop_words = set(stopwords.words('english'))
            
            def complexity_metrics(text):
                # Tokenize
                tokens = word_tokenize(text.lower())
                
                # Remove stopwords
                tokens = [token for token in tokens if token not in stop_words]
                
                return {
                    'unique_words': len(set(tokens)),
                    'stopword_ratio': len(tokens) / len(word_tokenize(text.lower())),
                    'special_char_ratio': len(re.findall(r'[^a-zA-Z\s]', text)) / len(text)
                }
            
            complexities = texts.apply(complexity_metrics)
            
            return {
                'avg_unique_words': complexities.apply(lambda x: x['unique_words']).mean(),
                'avg_stopword_ratio': complexities.apply(lambda x: x['stopword_ratio']).mean(),
                'avg_special_char_ratio': complexities.apply(lambda x: x['special_char_ratio']).mean()
            }
        
        return {
            'train_text_complexity': text_complexity(self.train_df['text']),
            'test_text_complexity': text_complexity(self.test_df['text'])
        }

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive HTML report
        """
        report = self.generate_quality_report()
        
        # Create HTML report
        html_report = f"""
        <html>
        <head>
            <title>Dataset Quality Assessment Report - {self.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }}
                h1, h2 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .critical {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Dataset Quality Assessment Report</h1>
            
            <h2>Dataset Overview</h2>
            <table>
                <tr><th>Dataset Name</th><td>{report['dataset_overview']['dataset_name']}</td></tr>
                <tr><th>Total Train Samples</th><td>{report['dataset_overview']['total_train_samples']}</td></tr>
                <tr><th>Total Test Samples</th><td>{report['dataset_overview']['total_test_samples']}</td></tr>
            </table>
            
            <h2>Missing Values</h2>
            <table>
                <tr><th>Train Missing</th><td>{report['missing_values']['total_missing_train']}</td></tr>
                <tr><th>Test Missing</th><td>{report['missing_values']['total_missing_test']}</td></tr>
            </table>
            
            <h2>Label Distribution</h2>
            <table>
                <tr><th>Train Label Balance</th><td>{report['label_distribution']['label_balance_train']:.2f}</td></tr>
                <tr><th>Test Label Balance</th><td>{report['label_distribution']['label_balance_test']:.2f}</td></tr>
            </table>
            
            <h2>Text Characteristics</h2>
            <table>
                <tr><th>Avg Train Text Length</th><td>{report['text_characteristics']['train_text_stats']['avg_length']:.2f}</td></tr>
                <tr><th>Avg Test Text Length</th><td>{report['text_characteristics']['test_text_stats']['avg_length']:.2f}</td></tr>
            </table>
            
            <h2>Duplicate Analysis</h2>
            <table>
                <tr><th>Train Duplicates</th><td>{report['duplicate_analysis']['train_total_duplicates']} ({report['duplicate_analysis']['train_duplicate_percentage']:.2f}%)</td></tr>
                <tr><th>Test Duplicates</th><td>{report['duplicate_analysis']['test_total_duplicates']} ({report['duplicate_analysis']['test_duplicate_percentage']:.2f}%)</td></tr>
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        with open('dataset_quality_report.html', 'w') as f:
            f.write(html_report)
        
        return report

# Example Usage
def main():
    # List of datasets to assess
    datasets = ['imdb', 'yelp_polarity', 'amazon_polarity']
    
    for dataset_name in datasets:
        print(f"\nAssessing {dataset_name} dataset:")
        quality_assessor = SentimentDatasetQualityAssessment(dataset_name)
        
        # Generate and print quality report
        report = quality_assessor.generate_comprehensive_report()
        
        # Print key findings
        print("\nKey Findings:")
        print("Total Samples:", report['dataset_overview']['total_train_samples'])
        print("Label Distribution:", report['label_distribution']['train_label_distribution'])
        print("Text Length (Avg):", report['text_characteristics']['train_text_stats']['avg_length'])
        print("Duplicates:", report['duplicate_analysis']['train_total_duplicates'])

if __name__ == "__main__":
    main()