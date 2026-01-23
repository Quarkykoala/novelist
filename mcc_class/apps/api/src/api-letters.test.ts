import { vi, describe, it, expect, beforeEach, beforeAll } from 'vitest';
import request from 'supertest';
import { Express } from 'express';

// Mocks must be defined before imports
const mockSelect = vi.fn();
const mockEq = vi.fn();
const mockRange = vi.fn();
const mockOrder = vi.fn();

// The builder object returned by calls
const mockBuilder = {
    select: mockSelect,
    eq: mockEq,
    range: mockRange,
    order: mockOrder,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    then: (resolve: any) => resolve({ data: [], error: null }),
};

// Chainable mocks return the builder
mockSelect.mockReturnValue(mockBuilder);
mockEq.mockReturnValue(mockBuilder);
mockRange.mockReturnValue(mockBuilder);
mockOrder.mockReturnValue(mockBuilder);

const mockFrom = vi.fn().mockReturnValue(mockBuilder);

// Mock the module
vi.mock('@supabase/supabase-js', () => ({
    createClient: () => ({
        from: mockFrom
    })
}));

describe('GET /api/letters', () => {
    let app: Express;

    beforeAll(async () => {
        // Set env vars before importing the app
        process.env.SUPABASE_URL = 'http://localhost:54321';
        process.env.SUPABASE_SERVICE_ROLE_KEY = 'mock-key';

        // Dynamic import to ensure env vars are set before module evaluation
        const mod = await import('./index');
        app = mod.app;
    });

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should query supabase with default pagination', async () => {
        const response = await request(app).get('/api/letters');

        expect(response.status).toBe(200);
        expect(mockFrom).toHaveBeenCalledWith('letters');
        expect(mockSelect).toHaveBeenCalledWith('*, departments(name), letter_tags(tags(name))');

        // Assert default range: page 1, limit 50 -> 0 to 49
        expect(mockRange).toHaveBeenCalledWith(0, 49);
    });

    it('should query supabase with custom pagination', async () => {
        const response = await request(app).get('/api/letters?page=2&limit=10');

        expect(response.status).toBe(200);
        // Page 2, limit 10 -> (2-1)*10 = 10 to 10+10-1 = 19
        expect(mockRange).toHaveBeenCalledWith(10, 19);
    });

    it('should query supabase with context filter', async () => {
        await request(app).get('/api/letters?context=TEST');
        expect(mockEq).toHaveBeenCalledWith('context', 'TEST');
        expect(mockRange).toHaveBeenCalledWith(0, 49); // Defaults still apply
    });

    it('should clamp negative page numbers to 1', async () => {
        await request(app).get('/api/letters?page=-5');
        expect(mockRange).toHaveBeenCalledWith(0, 49);
    });

    it('should cap limit at 100', async () => {
        await request(app).get('/api/letters?limit=1000');
        expect(mockRange).toHaveBeenCalledWith(0, 99);
    });
});
