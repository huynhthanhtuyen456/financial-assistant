const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// The URL of the page containing the PDF link.
const url = 'https://congbothongtin.ssc.gov.vn/';

async function downloadAllFinancialReports() {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({ 
    headless: false, // Set to true for production
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  }); 
  const page = await browser.newPage();

  // Set a realistic user agent
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

  console.log(`Navigating to ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle2' });
  console.log('Page loaded.');

  // Define the download path
  const downloadPath = path.resolve('./downloads');
  // Create downloads directory if it does not exist
  if (!fs.existsSync(downloadPath)) {
    fs.mkdirSync(downloadPath, { recursive: true });
  }

  console.log(`Setting download behavior to allow downloads in: ${downloadPath}`);

  // Set the download behavior using the Chrome DevTools Protocol
  const client = await page.target().createCDPSession();
  await client.send('Page.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: downloadPath,
  });

  try {
    // Wait for the page to fully load
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Get all download links from the current page
    console.log('Looking for PDF download links...');
    
    // Wait for the table to load
    await page.waitForSelector('table', { timeout: 10000 });
    
    // Get all PDF download elements with specific ID pattern (only <a> tags)
    const downloadElements = await page.evaluate(() => {
      const elements = [];
      
      // Look for <a> tags with ID pattern like "pt9:t1:0:cil4z"
      const allElements = document.querySelectorAll('a[id*="cil4z"]');
      
      allElements.forEach((element, index) => {
        // Find the parent row to get report name
        const row = element.closest('tr');
        if (row) {
          const cells = row.querySelectorAll('td');
          if (cells.length >= 5) {
            const reportName = cells[1] ? cells[1].textContent.trim() : `Report ${index + 1}`;
            elements.push({
              id: element.id,
              name: reportName,
              href: element.href,
              element: element
            });
          }
        }
      });
      
      return elements;
    });

    console.log(`Found ${downloadElements.length} PDF download elements`);

    if (downloadElements.length === 0) {
      console.log('No download elements found. Checking if we need to search...');
      
      // Try to search for all financial reports without specific stock code
      const searchButton = await page.$('input[type="submit"][value="Tìm kiếm"]');
      if (searchButton) {
        console.log('Clicking search button to get all results...');
        await searchButton.click();
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Retry getting download elements
        const retryElements = await page.evaluate(() => {
          const elements = [];
          const allElements = document.querySelectorAll('[id*="cil4z"]');
          
          allElements.forEach((element, index) => {
            const row = element.closest('tr');
            if (row) {
              const cells = row.querySelectorAll('td');
              if (cells.length >= 5) {
                const reportName = cells[1] ? cells[1].textContent.trim() : `Report ${index + 1}`;
                elements.push({
                  id: element.id,
                  name: reportName,
                  element: element
                });
              }
            }
          });
          
          return elements;
        });
        
        downloadElements.push(...retryElements);
        console.log(`Found ${downloadElements.length} total PDF download elements after search`);
      }
    }

    // Download all PDF files by clicking the <a> elements
    let downloadedCount = 0;
    
    // Remove duplicates based on ID
    const uniqueElements = downloadElements.filter((element, index, self) => 
      index === self.findIndex(e => e.id === element.id)
    );
    
    console.log(`Found ${uniqueElements.length} unique <a> elements to download`);
    
    for (let i = 0; i < uniqueElements.length; i++) {
      const element = uniqueElements[i];
      console.log(`\n[${i + 1}/${uniqueElements.length}] Downloading: ${element.name}`);
      
      try {
        // Find the <a> element by ID and click it with focus
        const clicked = await page.evaluate((elementId) => {
          const element = document.getElementById(elementId);
          if (element && element.tagName === 'A') {
            // Trigger focus and click as required
            element.focus();
            element.click();
            return true;
          }
          return false;
        }, element.id);
        
        if (clicked) {
          console.log(`✓ Clicked <a> element: ${element.name}`);
          downloadedCount++;
        } else {
          console.log(`✗ <a> element not found: ${element.name}`);
        }
        
        // Add delay between downloads to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 2000));
        
      } catch (error) {
        console.error(`✗ Failed to click <a> element ${element.name}:`, error.message);
      }
    }

    console.log(`\nDownload completed! Successfully clicked ${downloadedCount}/${downloadElements.length} download elements.`);
    console.log(`Files saved to: ${downloadPath}`);

  } catch (error) {
    console.error('An error occurred during the process:', error);
  }

  await browser.close();
  console.log('Browser closed.');
}

downloadAllFinancialReports();

