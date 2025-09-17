const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// The URL of the page containing the PDF link.
const url = 'https://congbothongtin.ssc.gov.vn/faces/NewsSearch';

async function downloadPdf() {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({ headless: true }); // Set to false to see the browser UI
  const page = await browser.newPage();

  // Set a realistic user agent
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');

  console.log(`Navigating to ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle2' });
  console.log('Page loaded.');

  // Define the download path
  const downloadPath = path.resolve('./downloads');
  // Create downloads directory if it does not exist
    if (!fs.existsSync(downloadPath)) {
        fs.mkdirSync(downloadPath);
    }

  console.log(`Setting download behavior to allow downloads in: ${downloadPath}`);

  // Set the download behavior using the Chrome DevTools Protocol
  const client = await page.target().createCDPSession();
  await client.send('Page.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: downloadPath,
  });

  try {
    // --- New logic for the new website ---
    const stockCode = 'FPT';
    const searchInputSelector = "input[id$=':it1::content']";
    const searchButtonSelector = "img[title='Tìm kiếm']";
    const resultLinkSelector = "a[href*='docId']"; // Selector for any link in the results

    console.log(`Typing stock code "${stockCode}" into the search box...`);
    await page.waitForSelector(searchInputSelector);
    await page.type(searchInputSelector, stockCode);

    console.log('Clicking the search button...');
    await page.waitForSelector(searchButtonSelector);
    await page.click(searchButtonSelector);

    console.log('Waiting for search results...');
    await page.waitForSelector(resultLinkSelector, { timeout: 20000 });
    console.log('Search results loaded.');
    
    // Find and click the first download link in the results table.
    const downloadLinkSelector = "table.af_table_data-table a[title$='.pdf']";
    await page.waitForSelector(downloadLinkSelector, { timeout: 10000 });
    console.log('Download link found. Clicking...');
    await page.click(downloadLinkSelector);

    console.log('Download initiated. Please wait for the file to be saved...');

    // Wait for the download to complete.
    await new Promise(resolve => setTimeout(resolve, 20000)); // Wait for 20 seconds

    console.log('PDF file should be downloaded.');

  } catch (error) {
    console.error('An error occurred during the process:', error);
  }


  await browser.close();
  console.log('Browser closed.');
}

downloadPdf();

